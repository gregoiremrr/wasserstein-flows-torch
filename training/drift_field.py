"""Sinkhorn (Wasserstein gradient flow) drift field and drift loss.

Implements the training objective of "One-Step Generative Modeling via
Wasserstein Gradient Flows" (W-Flow, Han et al., 2026). The generated
distribution q evolves along the steepest-descent direction of the (debiased)
Sinkhorn divergence S_eps(q, p) in Wasserstein space. The induced velocity
field is the difference of two entropic-OT barycentric projections:

    V(x) = T^eps_{q,p}(x) - T^eps_{q,q'}(x)      (Eq. 10 / Eq. 13)

where T^eps_{q,p} transports the generated batch toward the real batch
(attraction), and T^eps_{q,q'} is the *debiased self-transport* estimated from a
second, independent generated batch q' (the "two-batch" estimator of Sec. 3.3,
which replaces the diagonal-masking heuristic of the Drifting model). The loss
then pushes each generated sample toward its frozen drifted target
``stopgrad(x + V)``, exactly mirroring the drift objective.

This is the debiased entropic-OT method only (no MMD / KL ablation variants).
The calling convention deliberately mirrors the doubly-normalized-softmax
``drift_loss`` of the Drifting model so the loss orchestration is a drop-in
swap: the only conceptual change is that softmax affinities become Sinkhorn
transport plans, and the in-batch self term becomes an independent second batch.

All tensors operate per "row group" B: each group holds the generated, negative
(second generated batch) and positive samples that interact with one another.
In the generator training loop, B = (#class_labels x #feature_locations), N is
the number of samples in a group, and D is the feature dimensionality.
"""

import math

import torch

#----------------------------------------------------------------------------
# Batched pairwise Euclidean distance.
# x: [B, N, D], y: [B, M, D] -> [B, N, M]

def cdist(x, y, eps=1e-8):
    xydot = torch.einsum('bnd,bmd->bnm', x, y)
    xnorms = torch.einsum('bnd,bnd->bn', x, x)
    ynorms = torch.einsum('bmd,bmd->bm', y, y)
    sq_dist = xnorms[:, :, None] + ynorms[:, None, :] - 2 * xydot
    return torch.sqrt(sq_dist.clamp(min=eps))

#----------------------------------------------------------------------------
# Balanced entropic-OT plan via Sinkhorn-Knopp iterations (log domain).
#
# C: [B, N, M] cost matrix; reg: entropic regularisation eps (> 0).
# target_weights: optional non-uniform target marginal [B, M] (else uniform).
# Returns the transport plan Pi: [B, N, M] with row marginal 1/N.

@torch.no_grad()
def _sinkhorn_plan(C, reg, num_iter, target_weights=None):
    B, N, M = C.shape
    device, dtype = C.device, C.dtype
    logK = -C / reg

    log_a = torch.full((B, N), -math.log(N), device=device, dtype=dtype)
    if target_weights is None:
        log_b = torch.full((B, M), -math.log(M), device=device, dtype=dtype)
    else:
        b = target_weights.to(device=device, dtype=dtype)
        b = b / b.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        log_b = torch.log(b.clamp_min(1e-30))

    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    for _ in range(max(int(num_iter), 1)):
        log_u = log_a - torch.logsumexp(logK + log_v[:, None, :], dim=-1)
        log_v = log_b - torch.logsumexp(logK.transpose(1, 2) + log_u[:, None, :], dim=-1)
    return torch.exp(log_u[:, :, None] + logK + log_v[:, None, :])

#----------------------------------------------------------------------------
# Entropic-OT barycentric projection  T^eps(x) = (Pi @ support) / row_mass.
# Optimal transport runs without gradients; gradients flow only through the
# generated features when the projection target is subtracted in the loss.

@torch.no_grad()
def _barycentric_map(x, support, reg, num_iter, diag_mask=False,
                     target_weights=None, use_quadratic_cost=True):
    C = cdist(x, support)
    if use_quadratic_cost:
        C = 0.5 * C * C
    if diag_mask:
        n = min(x.shape[1], support.shape[1])
        idx = torch.arange(n, device=x.device)
        C[:, idx, idx] = C[:, idx, idx] + 1e6
    Pi = _sinkhorn_plan(C, reg, num_iter, target_weights=target_weights)
    row_mass = Pi.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.bmm(Pi, support) / row_mass

#----------------------------------------------------------------------------
# Debiased Sinkhorn drift loss for a single feature.
#
# Args:
#     gen:        [B, C_g, D] generated samples (with gradient).
#     fixed_pos:  [B, C_p, D] positive (real, same-class) samples.
#     fixed_neg:  [B, C_n, D] second, independent generated batch q' used for
#                 the debiased self-transport term. May be None (then the
#                 self term falls back to gen itself with diagonal masking).
#     fixed_uncond: [B, C_u, D] unconditional reals for velocity-guidance CFG.
#                 May be None (no guidance).
#     weight_neg:  optional per-sample marginal weights for q' [B, C_n].
#     cfg_weight:  per-group guidance scale w >= 0 [B] (only with fixed_uncond).
#     R_list:      entropic-regularisation values eps; one debiased velocity per
#                  eps, each normalised then summed (a single value is the
#                  default and matches the paper's single-eps setting).
#     sinkhorn_num_iter: Sinkhorn-Knopp iterations per OT problem.
#     use_quadratic_cost: c(x, y) = ||x - y||^2 / 2 (paper default) vs ||x - y||.
#     disable_diag_mask:  skip diagonal masking of the self term (correct when
#                  fixed_neg is an *independent* second batch -- two-batch mode).
#
# Returns:
#     loss: [B] per-group MSE between the generated feature and its frozen
#           drifted target, computed in the normalized feature space.
#     info: dict of scalars (feature scale + per-eps drift magnitude).

def drift_loss_ot(
    gen,
    fixed_pos,
    fixed_neg=None,
    weight_gen=None,
    weight_pos=None,
    weight_neg=None,
    fixed_uncond=None,
    cfg_weight=None,
    R_list=(0.05,),
    sinkhorn_num_iter=10,
    use_quadratic_cost=True,
    disable_diag_mask=True,
):
    B, C_g, S = gen.shape

    if fixed_neg is None:
        # Fall back to a single-batch self term (gen vs gen, diagonal masked).
        fixed_neg = gen.detach()
        disable_diag_mask = False
    C_n = fixed_neg.shape[1]

    if weight_neg is None:
        weight_neg = torch.ones_like(fixed_neg[:, :, 0])

    gen = gen.float()
    fixed_pos = fixed_pos.float()
    fixed_neg = fixed_neg.float()
    weight_neg = weight_neg.float()
    use_cfg = fixed_uncond is not None and cfg_weight is not None
    if use_cfg:
        fixed_uncond = fixed_uncond.float()

    old_gen = gen.detach()

    # The entire target (goal) is computed without gradients.
    with torch.no_grad():
        info = {}

        # -- feature-scale normalization (same convention as the softmax drift) --
        # Distances are made order-1 so a fixed eps has consistent meaning
        # across features of different magnitude / dimensionality.
        targets = torch.cat([old_gen, fixed_neg, fixed_pos], dim=1)
        targets_w = torch.cat([
            torch.ones_like(old_gen[:, :, 0]), weight_neg,
            torch.ones_like(fixed_pos[:, :, 0]),
        ], dim=1)
        dist = cdist(old_gen, targets)
        # Match the reference: with a quadratic cost the relevant magnitude is
        # the RMS distance (sqrt(mean(d^2))), so eps -- which multiplies the
        # squared cost -- stays order-1; with the linear cost it's the mean
        # distance. (Numerically close in high dim, but kept consistent.)
        if use_quadratic_cost:
            weighted_dist_sq = (dist * dist) * targets_w[:, None, :]
            scale = (weighted_dist_sq.mean() / targets_w.mean().clamp_min(1e-8)).sqrt()
        else:
            weighted_dist = dist * targets_w[:, None, :]
            scale = weighted_dist.mean() / targets_w.mean().clamp_min(1e-8)
        info['scale'] = scale
        scale_inputs = (scale / (S ** 0.5)).clamp(min=1e-3)

        old_gen_scaled = old_gen / scale_inputs
        pos_scaled = fixed_pos / scale_inputs
        neg_scaled = fixed_neg / scale_inputs
        uncond_scaled = (fixed_uncond / scale_inputs) if use_cfg else None

        # Reg scales with cost magnitude: S for quadratic cost, sqrt(S) for L2.
        # (||a - b||^2 grows ~ S * scale_inputs^2 ~ S in normalized coords.)
        reg_scale = float(S) if use_quadratic_cost else math.sqrt(S)
        bary_kw = dict(num_iter=sinkhorn_num_iter, use_quadratic_cost=use_quadratic_cost)

        # -- accumulate debiased velocity across eps values --
        force_across_R = torch.zeros_like(old_gen_scaled)
        for R in R_list:
            reg = old_gen_scaled.new_tensor(float(R) * reg_scale)

            # Attraction toward the real batch.
            T_pq = _barycentric_map(old_gen_scaled, pos_scaled, reg, diag_mask=False, **bary_kw)
            # Debiased self-transport against the independent second batch q'.
            T_qq = _barycentric_map(
                old_gen_scaled, neg_scaled, reg,
                diag_mask=(not disable_diag_mask),
                target_weights=weight_neg, **bary_kw,
            )
            V_raw = T_pq - T_qq

            # Velocity-guidance CFG (Eq. 16): add w * (T_{q,p} - T_{q,uncond}).
            if use_cfg:
                T_qu = _barycentric_map(old_gen_scaled, uncond_scaled, reg, diag_mask=False, **bary_kw)
                V_raw = V_raw + cfg_weight.view(-1, 1, 1) * (T_pq - T_qu)

            f_norm_val = (V_raw ** 2).mean()
            info[f'loss_{R}'] = f_norm_val

            # Drift normalization per eps, then sum.
            force_scale = f_norm_val.clamp(min=1e-8).sqrt()
            force_across_R = force_across_R + V_raw / force_scale

        goal_scaled = old_gen_scaled + force_across_R

    gen_scaled = gen / scale_inputs
    diff = gen_scaled - goal_scaled
    loss = (diff ** 2).mean(dim=(-1, -2))
    return loss, info

#----------------------------------------------------------------------------
