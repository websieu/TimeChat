apt install ffmpeg -y
conda env create -f environment.yml
conda activate timechat
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

### download model ###
mkdir ckpt && cd ckpt
mkdir eva-vit-g && cd eva-vit-g
wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth
cd ..
mkdir instruct-blip && cd instruct-blip
wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth
cd ..

### download timechat ###

sudo apt-get install git-lfs
git-lfs install
git clone https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-7B-Finetuned

