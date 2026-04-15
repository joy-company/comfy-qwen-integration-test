"""
Configuration for Qwen 9B Agent + ComfyUI + Gemini integration.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ComfyUI API Server
COMFYUI_HOST = "127.0.0.1"
COMFYUI_PORT = 8188
COMFYUI_BASE_URL = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"

# Qwen 9B via Ollama (OpenAI-compatible endpoint)
QWEN_BASE_URL = "http://127.0.0.1:11434/v1"
QWEN_MODEL = "qwen3:8b"

# Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"

# Default workflow file
DEFAULT_WORKFLOW_PATH = "flux_depth_lora_example.json"

# Polling interval for image generation status (seconds)
POLL_INTERVAL = 2.0
MAX_POLL_ATTEMPTS = 300  # 10 minutes max wait

# Episode pipeline
MAX_REVIEW_RETRIES = 3  # Max Gemini prompt revisions before forcing approval
