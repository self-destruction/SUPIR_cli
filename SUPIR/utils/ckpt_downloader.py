import os
import time
import requests

def download_checkpoint(model_url, model_name, ckpt_dir):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    checkpoint_path = os.path.join(ckpt_dir, f"{model_name}.safetensors")

    if os.path.exists(checkpoint_path):
        return f"Checkpoint '{model_name}' already exists in {ckpt_dir}."

    try:
        response = requests.get(model_url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        total_downloaded = 0
        start_time = time.time()

        with open(checkpoint_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                total_downloaded += len(chunk)
                elapsed_time = time.time() - start_time
                if elapsed_time == 0:
                    elapsed_time = 0.1  # avoid division by zero by setting a minimum elapsed time
                speed = (total_downloaded / elapsed_time) / (1024 ** 2)  # convert bytes/sec to MB/sec
                done = int(50 * total_downloaded / total_size)
                print(f"\rDownloading {model_name}: [{'#' * done}{'.' * (50 - done)}] {total_downloaded * 100 / total_size:.2f}% @ {speed:.2f} MB/s", end='', flush=True)

        print("\nDownload completed successfully.")
        return f"Checkpoint '{model_name}' downloaded successfully to {ckpt_dir}."
    except requests.exceptions.RequestException as e:
        return f"Error downloading checkpoint: {str(e)}"

def download_checkpoint_handler(model_choice, ckpt_dir):
    model_mapping = {
        "SDXL 1.0 Base": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0_0.9vae.safetensors",
        "RealVisXL_V4": "https://huggingface.co/SG161222/RealVisXL_V4.0/resolve/main/RealVisXL_V4.0.safetensors",
        "Animagine XL V3.1": "https://civitai.com/api/download/models/403131?type=Model&format=SafeTensor&size=full&fp=fp16&token=5577db242d28f46030f55164cdd2da5d",
		"Juggernaut XL V10" : "https://huggingface.co/RunDiffusion/Juggernaut-X-v10/resolve/main/Juggernaut-X-RunDiffusion-NSFW.safetensors"
    }

    if model_choice in model_mapping:
        model_url = model_mapping[model_choice]
        return download_checkpoint(model_url, model_choice, ckpt_dir)
    else:
        return "Invalid model choice."
