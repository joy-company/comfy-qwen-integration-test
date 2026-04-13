"""
Download required models for Flux Depth LoRA workflow.

This script downloads models from HuggingFace and organizes them into the ComfyUI directory structure.

Usage:
    python download_models.py --comfyui-path "C:/path/to/ComfyUI"

Required:
    - huggingface_hub: pip install huggingface_hub
    - ~20GB free disk space
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("[ERROR] huggingface_hub is not installed.")
    print("        Install it with: pip install huggingface_hub")
    sys.exit(1)

# Model configuration: (repo_id, filename, local_path_relative_to_models)
MODELS_TO_DOWNLOAD = [
    # Base model
    ("black-forest-labs/FLUX.1-dev", "flux1-dev-fp8.safetensors", "diffusion_models"),
    
    # LoRA
    ("black-forest-labs/FLUX.1-Depth-dev", "flux1-depth-dev-lora.safetensors", "loras"),
    
    # VAE
    ("black-forest-labs/FLUX.1-dev", "ae.safetensors", "vae"),
    
    # Text Encoders
    ("black-forest-labs/FLUX.1-dev", "t5xxl_fp16.safetensors", "text_encoders"),
    ("black-forest-labs/FLUX.1-dev", "clip_l.safetensors", "text_encoders"),
    
    # Depth Model
    ("IDEA-Research/lotus-depth-v1", "lotus-depth-d-v1-1.safetensors", "diffusion_models"),
]


def download_models(comfyui_path: str):
    """Download and organize models."""
    comfyui_path = Path(comfyui_path)
    models_dir = comfyui_path / "models"
    
    # Check for placeholder path patterns (more specific)
    path_str = str(comfyui_path).lower()
    if path_str.startswith("c:\\path") or path_str.startswith("c:/path") or \
       (path_str.count("path") >= 2) or path_str.endswith("comfyui\"") or \
       "\\to\\" in path_str or "/to/" in path_str:
        print("[ERROR] Please provide your actual ComfyUI path")
        print(f"        You provided: {comfyui_path}")
        print()
        print("        Example:")
        print('          python download_models.py --comfyui-path "C:/ComfyUI"')
        print('          python download_models.py --comfyui-path "D:/AI/ComfyUI"')
        return False
    
    if not comfyui_path.exists():
        print(f"[ERROR] ComfyUI path does not exist: {comfyui_path}")
        print()
        print("        Please check that:")
        print("        1. ComfyUI is installed at the specified path")
        print("        2. The path is correct (use forward slashes or escape backslashes)")
        print()
        print("        Example:")
        print('          python download_models.py --comfyui-path "C:/ComfyUI"')
        return False
    
    print(f"[INFO] ComfyUI path: {comfyui_path}")
    print(f"[INFO] Models directory: {models_dir}")
    print()
    
    for repo_id, filename, subfolder in MODELS_TO_DOWNLOAD:
        local_dir = models_dir / subfolder
        local_dir.mkdir(parents=True, exist_ok=True)
        
        local_path = local_dir / filename
        
        if local_path.exists():
            print(f"[OK] {filename} (already exists)")
            continue
        
        print(f"[DOWNLOAD] {filename}")
        print(f"           From: {repo_id}")
        print(f"           To: {local_dir}")
        
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            )
            print(f"[OK] {filename}")
        except Exception as e:
            error_msg = str(e).lower()
            if "401" in str(e) or "gated" in error_msg or "access" in error_msg or "authenticate" in error_msg:
                print(f"[AUTH_ERROR] {filename}")
                print(f"             This model requires HuggingFace authentication.")
                print()
                print("             Steps to fix:")
                print("             1. Create a HuggingFace account: https://huggingface.co/join")
                print("             2. Accept the model terms:")
                print(f"                - {repo_id}")
                print("                (visit the repo and click 'Access repository')")
                print()
                print("             3. Get your access token:")
                print("                - Go to: https://huggingface.co/settings/tokens")
                print("                - Click 'New token' (role: read)")
                print("                - Copy the token")
                print()
                print("             4. Authenticate:")
                print("                huggingface-cli login")
                print("                (paste your token when prompted)")
                print()
                print("             5. Re-run this script")
                return False
            else:
                print(f"[ERROR] Failed to download {filename}: {e}")
                return False
        
        print()
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download models for Flux Depth LoRA workflow"
    )
    parser.add_argument(
        "--comfyui-path",
        type=str,
        required=True,
        help="Path to ComfyUI directory (e.g., C:/ComfyUI)",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Flux Depth LoRA Model Downloader")
    print("=" * 60)
    print()
    print("[WARNING] This will download ~20GB of models.")
    print("          Ensure you have sufficient disk space and network bandwidth.")
    print()
    
    success = download_models(args.comfyui_path)
    
    if success:
        print("=" * 60)
        print("[SUCCESS] All models downloaded successfully!")
        print("=" * 60)
    else:
        print("=" * 60)
        print("[FAILED] Download failed!")
        print("=" * 60)


if __name__ == "__main__":
    main()
