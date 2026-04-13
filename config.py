"""
Configuration for Qwen 9B Agent + ComfyUI integration.
"""

# ComfyUI API Server
COMFYUI_HOST = "127.0.0.1"
COMFYUI_PORT = 8188
COMFYUI_BASE_URL = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"

# Qwen 9B via Ollama (OpenAI-compatible endpoint)
QWEN_BASE_URL = "http://127.0.0.1:11434/v1"
QWEN_MODEL = "qwen3:8b"

# Default workflow file
DEFAULT_WORKFLOW_PATH = "flux_depth_lora_example.json"

# Polling interval for image generation status (seconds)
POLL_INTERVAL = 2.0
MAX_POLL_ATTEMPTS = 300  # 10 minutes max wait
