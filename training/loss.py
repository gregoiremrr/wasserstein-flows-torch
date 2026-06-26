"""W-Flow drift loss orchestration.

Given a batch of class labels with cached positive (same-class) and
unconditional real samples, this computes the multi-feature Sinkhorn drift loss
of "One-Step Generative Modeling via Wasserstein Gradient Flows":

  1. Extract features (a raw "global" feature plus multi-scale MAE features) for
     the positive + unconditional samples (no gradient).
  2. Generate `gen_per_label` samples per label and extract their features (with
     gradient through the generator and the feature encoder).
  3. Generate a SECOND, independent batch of `gen_per_label` samples per label
     (stop-gradient): the two-batch estimator for the debiased self-transport
     term T^eps_{q,q'} (Sec. 3.3), which replaces the diagonal-masking heuristic.
  4. For each feature (per scale / per spatial location) compute the debiased
     Sinkhorn drift loss, summing across features.

Classifier-free guidance is realized at training time only via *velocity
guidance* (Eq. 16): each label samples a CFG scale `alpha = w + 1` from a
power-law distribution, the unconditional reals induce an extra
``w * (T_{q,p} - T_{q,uncond})`` term in the velocity, and `alpha` is fed to the
generator as a conditioning input. Inference stays one-step (1-NFE).
"""

import torch
from torch_utils import persistence
from training.drift_field import drift_loss_ot

#----------------------------------------------------------------------------

@persistence.persistent_class
class DriftLoss:
    def __init__(
        self,
        gen_per_label=64,
        self_gen_per_label=None,    # Second-batch size for self-transport; None => same as gen_per_label.
        cfg_min=1.0,
        cfg_max=4.0,
        neg_cfg_pw=1.0,
        no_cfg_frac=0.0,
        R_list=(0.05,),             # Entropic-regularisation eps values (single value by default).
        sinkhorn_num_iter=10,
        use_quadratic_cost=True,
        disable_diag_mask=True,
        activation_kwargs=None,
    ):
        self.gen_per_label = gen_per_label
        self.self_gen_per_label = self_gen_per_label if self_gen_per_label is not None else gen_per_label
        self.cfg_min = cfg_min
        self.cfg_max = cfg_max
        self.neg_cfg_pw = neg_cfg_pw
        self.no_cfg_frac = no_cfg_frac
        self.R_list = tuple(R_list)
        self.sinkhorn_num_iter = int(sinkhorn_num_iter)
        self.use_quadratic_cost = bool(use_quadratic_cost)
        self.disable_diag_mask = bool(disable_diag_mask)
        if activation_kwargs is None:
            activation_kwargs = dict(
                patch_mean_size=[2, 4], patch_std_size=[2, 4],
                use_std=True, use_mean=True, every_k_block=2,
            )
        self.activation_kwargs = dict(activation_kwargs)

    # -- CFG-scale sampling (power-law p(alpha) ~ alpha^-neg_cfg_pw) ----------

    def sample_cfg(self, n, device, generator=None):
        frac = torch.rand(n, device=device, generator=generator)
        pw = 1.0 - self.neg_cfg_pw
        cmin, cmax = self.cfg_min, self.cfg_max
        if abs(pw) < 1e-6:
            import math
            cfg = torch.exp(math.log(cmin) + frac * (math.log(cmax) - math.log(cmin)))
        else:
            cfg = (cmin ** pw + frac * (cmax ** pw - cmin ** pw)) ** (1.0 / pw)
        if self.no_cfg_frac > 0:
            frac2 = torch.rand(n, device=device, generator=generator)
            cfg = torch.where(frac2 < self.no_cfg_frac, torch.ones_like(cfg), cfg)
        return cfg

    # -- feature extraction --------------------------------------------------

    def _features(self, feature_encoder, x):
        # Flatten in BHWC order to match the reference JAX layout. (Numerically a
        # consistent permutation of the feature axis, which the drift loss is
        # invariant to, but kept for 1:1 fidelity with the reference.)
        out = {'global': x.permute(0, 2, 3, 1).reshape(x.shape[0], 1, -1)}
        if feature_encoder is not None:
            out.update(feature_encoder.get_activations(x, **self.activation_kwargs))
        return out

    @staticmethod
    def _group_by_token(feat):
        # [B, N, T, D] -> [(B*T), N, D]
        B, N, T, D = feat.shape
        return feat.permute(0, 2, 1, 3).reshape(B * T, N, D)

    # -- main call -----------------------------------------------------------

    def __call__(self, model, feature_encoder, labels, pos_images, uncond_images, cfg):
        """
        Args:
            model: generator (possibly DDP-wrapped); ``model(class_idx, cfg)`` -> images.
            feature_encoder: frozen MAE (or None for raw-pixel-only drift).
            labels: [Nc] long class indices.
            pos_images: [Nc, n_pos, C, H, W] positive reals.
            uncond_images: [Nc, n_uncond, C, H, W] unconditional reals (velocity-CFG).
            cfg: [Nc] per-label CFG scale alpha = w + 1.
        """
        Nc, n_pos = pos_images.shape[0], pos_images.shape[1]
        n_uncond = uncond_images.shape[1]
        g = self.gen_per_label
        gs = self.self_gen_per_label
        use_cfg = n_uncond > 0

        # Frozen features for positives + unconditional negatives.
        n_real_block = n_pos + n_uncond
        real_input = torch.cat([pos_images, uncond_images], dim=1)
        real_input = real_input.reshape(Nc * n_real_block, *real_input.shape[2:])
        with torch.no_grad():
            real_feats = self._features(feature_encoder, real_input)
        real_feats = {k: v.reshape(Nc, n_real_block, *v.shape[1:]) for k, v in real_feats.items()}

        # Generated samples (with gradient).
        class_idx = labels.repeat_interleave(g)
        cfg_rep = cfg.repeat_interleave(g)
        gen_images = model(class_idx, cfg_rep)
        gen_feats = self._features(feature_encoder, gen_images)
        gen_feats = {k: v.reshape(Nc, g, *v.shape[1:]) for k, v in gen_feats.items()}

        # Second, independent generated batch (stop-gradient) for the debiased
        # self-transport term T^eps_{q,q'} (two-batch estimator).
        with torch.no_grad():
            self_class_idx = labels.repeat_interleave(gs)
            self_cfg_rep = cfg.repeat_interleave(gs)
            self_images = model(self_class_idx, self_cfg_rep)
            self_feats = self._features(feature_encoder, self_images)
            self_feats = {k: v.reshape(Nc, gs, *v.shape[1:]) for k, v in self_feats.items()}

        total_loss = gen_images.new_zeros(())
        stats = {}
        for key, gen_f in gen_feats.items():
            real = real_feats[key]                       # [Nc, n_real_block, T, D]
            pos_f = real[:, :n_pos]
            uncond_f = real[:, n_pos:]
            T = gen_f.shape[2]

            gen_r = self._group_by_token(gen_f)          # [(Nc*T), g, D]
            pos_r = self._group_by_token(pos_f)
            self_r = self._group_by_token(self_feats[key])  # [(Nc*T), gs, D]

            uncond_r = None
            cfg_w = None
            if use_cfg:
                uncond_r = self._group_by_token(uncond_f)
                cfg_w = (cfg - 1.0).repeat_interleave(T)  # w = alpha - 1, grouped per token

            loss_k, info_k = drift_loss_ot(
                gen=gen_r, fixed_pos=pos_r, fixed_neg=self_r,
                fixed_uncond=uncond_r, cfg_weight=cfg_w,
                R_list=self.R_list, sinkhorn_num_iter=self.sinkhorn_num_iter,
                use_quadratic_cost=self.use_quadratic_cost,
                disable_diag_mask=self.disable_diag_mask,
            )
            total_loss = total_loss + loss_k.mean()
            stats[f'drift/{key}'] = loss_k.mean().detach()

        out_stats = dict(loss=total_loss.detach(), cfg=cfg.mean().detach(),
                         gen_images=gen_images.detach(), gen_labels=class_idx.detach())
        return total_loss, out_stats

#----------------------------------------------------------------------------
