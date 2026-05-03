conda create -n open_unlearning python=3.10 -y
conda activate open_unlearning

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

pip install psutil ninja packaging wheel setuptools
pip install cmake

pip install flash-attn==2.8.3 --no-build-isolation

pip install -r requirements_2026.txt