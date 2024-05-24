
### Dependencies and Installation

```bash
pip install -q -r requirements.txt
pip install -q --pre xformers==0.0.24
pip install -q torchvision==0.16.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -q torch==2.2.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```
  
### Download Checkpoints

```bash
lightning_model = "https://huggingface.co/RunDiffusion/Juggernaut-XL-Lightning/resolve/main/Juggernaut_RunDiffusionPhoto2_Lightning_4Steps.safetensors"
!wget -c {lightning_model} -P {WORK_DIR}/SUPIR_cli/models

vit_model = "https://huggingface.co/camenduru/SUPIR/resolve/main/clip-vit-large-patch14-336.tar"
!wget {vit_model} -P {WORK_DIR}/models/

vit_large_model = "https://huggingface.co/camenduru/SUPIR/resolve/main/clip-vit-large-patch14.tar"
!wget {vit_large_model} -P {WORK_DIR}/models/

supir_model = "https://huggingface.co/camenduru/SUPIR/resolve/main/SUPIR-v0Q.ckpt"
!wget -c {supir_model} -O "{WORK_DIR}/SUPIR_cli/models/v0Q.ckpt"

pytorch_model = "https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/resolve/main/open_clip_pytorch_model.bin"
!wget -c {pytorch_model} -P {WORK_DIR}/SUPIR_cli/models/

!tar -xvf {WORK_DIR}/models/clip-vit-large-patch14-336.tar -C {WORK_DIR}/SUPIR_cli/models/clip-vit-large-patch14-336
!rm -rf {WORK_DIR}/models/clip-vit-large-patch14-336.tar
!tar -xvf /kaggle/temp/models/clip-vit-large-patch14.tar -C {WORK_DIR}/SUPIR_cli/models/clip-vit-large-patch14
!rm -rf {WORK_DIR}/models/clip-vit-large-patch14.tar
```

### Run of SUPIR  

```bash
python -W ignore::UserWarning: script_cli.py --img_path "{FILE_FOR_ENHANCE}" --save_dir "{OUTPUT_DIR}/Supir_output" --upscale 2 --fast_load_sd --no_llava --options SUPIR_v0_Juggernautv9_lightning_tiled --ae_dtype fp32 --sampler DPMPP2M --use_tile_vae
```
