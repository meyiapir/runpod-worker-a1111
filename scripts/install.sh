#!/usr/bin/env bash
echo "Deleting Automatic1111 Web UI"
rm -rf /workspace/stable-diffusion-webui

echo "Deleting venv"
rm -rf /workspace/venv

echo "Cloning A1111 repo to /workspace"
cd /workspace
git clone --depth=1 https://github.com/AUTOMATIC1111/stable-diffusion-webui.git

echo "Installing Ubuntu updates"
apt update
apt -y upgrade

echo "Creating and activating venv"
cd stable-diffusion-webui
python3 -m venv /workspace/venv
source /workspace/venv/bin/activate

echo "Installing Torch"
pip3 install --no-cache-dir torch==2.0.1+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Installing xformers"
pip3 install --no-cache-dir xformers==0.0.22

echo "Installing A1111 Web UI"
wget https://raw.githubusercontent.com/ashleykleynhans/runpod-worker-a1111/main/install-automatic.py
python3 -m install-automatic --skip-torch-cuda-test

echo "Cloning ControlNet extension repo"
cd /workspace/stable-diffusion-webui
git clone --depth=1 https://github.com/Mikubill/sd-webui-controlnet.git extensions/sd-webui-controlnet

echo "Cloning the ReActor extension repo"
git clone --depth=1 https://github.com/Gourieff/sd-webui-reactor.git extensions/sd-webui-reactor

echo "Cloning the After Detailer extension repo"
git clone --depth=1 https://github.com/Bing-su/adetailer.git extensions/adetailer

echo "Cloning A Person Mask Generator repo"
git clone --depth=1 https://github.com/djbielejeski/a-person-mask-generator.git extensions/a-person-mask-generator

echo "Installing dependencies for A Person Mask Generator"
cd /workspace/stable-diffusion-webui/extensions/a-person-mask-generator
pip3 install -r requirements.txt

echo "Installing dependencies for ControlNet"
cd /workspace/stable-diffusion-webui/extensions/sd-webui-controlnet
pip3 install -r requirements.txt

echo "Installing dependencies for ReActor"
cd /workspace/stable-diffusion-webui/extensions/sd-webui-reactor
pip3 install -r requirements.txt
pip3 install onnxruntime-gpu

echo "Installing dependencies for After Detailer"
cd /workspace/stable-diffusion-webui/extensions/adetailer
python3 -m install

echo "Installing the model for ReActor"
mkdir -p /workspace/stable-diffusion-webui/models/insightface
cd /workspace/stable-diffusion-webui/models/insightface
wget https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx

echo "Configuring ReActor to use the GPU instead of CPU"
echo "CUDA" > /workspace/stable-diffusion-webui/extensions/sd-webui-reactor/last_device.txt

echo "Installing RunPod Serverless dependencies"
cd /workspace/stable-diffusion-webui
pip3 install huggingface_hub runpod

echo "Downloading Deliberate v2 model"
cd /workspace/stable-diffusion-webui/models/Stable-diffusion
wget -O deliberate_v2.safetensors https://huggingface.co/XpucT/Deliberate/resolve/main/Deliberate_v2.safetensors

echo "Downloading Realistic Vision Inpaint v6 model"
cd /workspace/stable-diffusion-webui/models/Stable-diffusion
wget -O realisticVisionV60B1_v60B1InpaintingVAE.safetensors https://civitai.com/api/download/models/245627

echo "Downloading SDXL base model"
wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors

echo "Downloading SDXL Refiner"
wget https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors

echo "Downloading SD 1.5 VAE"
cd /workspace/stable-diffusion-webui/models/VAE
wget https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors

echo "Downloading SDXL VAE"
wget https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors

echo "Downloading SD 1.5 ControlNet models"
mkdir -p /workspace/stable-diffusion-webui/models/ControlNet
cd /workspace/stable-diffusion-webui/models/ControlNet
wget https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth
wget https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.pth
wget https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth
wget https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_inpaint.pth
wget https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart.pth
wget https://huggingface.co/ioclab/ioc-controlnet/resolve/main/models/control_v1p_sd15_brightness.safetensors
wget https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1e_sd15_tile.pth

echo "Downloading SDXL ControlNet models"
wget https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/diffusers_xl_canny_full.safetensors

echo "Downloading InstantID ControlNet models"
wget -O ip-adapter_instant_id_sdxl.bin "https://huggingface.co/InstantX/InstantID/resolve/main/ip-adapter.bin?download=true"
wget -O control_instant_id_sdxl.safetensors"https://huggingface.co/InstantX/InstantID/resolve/main/ControlNetModel/diffusion_pytorch_model.safetensors?download=true"

echo "Downloading Upscalers"
mkdir -p /workspace/stable-diffusion-webui/models/ESRGAN
cd /workspace/stable-diffusion-webui/models/ESRGAN
wget https://huggingface.co/ashleykleynhans/upscalers/resolve/main/4x-UltraSharp.pth
wget https://huggingface.co/ashleykleynhans/upscalers/resolve/main/lollypop.pth

echo "Creating log directory"
mkdir -p /workspace/logs

echo "Installing config files"
cd /workspace/stable-diffusion-webui
rm webui-user.sh config.json ui-config.json
wget https://raw.githubusercontent.com/ashleykleynhans/runpod-worker-a1111/main/webui-user.sh
wget https://raw.githubusercontent.com/ashleykleynhans/runpod-worker-a1111/main/config.json
wget https://raw.githubusercontent.com/ashleykleynhans/runpod-worker-a1111/main/ui-config.json

echo "Starting A1111 Web UI"
deactivate
export HF_HOME="/workspace"
cd /workspace/stable-diffusion-webui
./webui.sh -f
