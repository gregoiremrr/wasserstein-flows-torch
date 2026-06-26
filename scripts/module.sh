python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy ipykernel click Pillow psutil requests scipy tqdm wandb matplotlib
pip install --upgrade torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
