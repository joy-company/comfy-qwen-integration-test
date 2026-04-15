"""
ComfyUI HTTP API client.

Handles communication with the ComfyUI server:
- Queue prompts (workflow execution)
- Poll for completion
- Retrieve generated images
- Query available models (LoRAs, checkpoints, etc.)
"""

import json
import time
import uuid
import urllib.request
import urllib.error
from pathlib import Path

from config import COMFYUI_BASE_URL, POLL_INTERVAL, MAX_POLL_ATTEMPTS


class ComfyUIClient:
    def __init__(self, base_url: str = COMFYUI_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.client_id = str(uuid.uuid4())

    # ------------------------------------------------------------------
    # Low-level HTTP helpers (stdlib only, no extra dependencies)
    # ------------------------------------------------------------------

    def _get(self, path: str) -> dict:
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())

    def _post_json(self, path: str, data: dict) -> dict:
        url = f"{self.base_url}{path}"
        payload = json.dumps(data).encode()
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            raise RuntimeError(
                f"HTTP {e.code} from {path}: {body}"
            ) from e

    def _get_bytes(self, path: str) -> bytes:
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_alive(self) -> bool:
        """Check if the ComfyUI server is reachable."""
        try:
            self._get("/system_stats")
            return True
        except Exception:
            return False

    def queue_prompt(self, api_workflow: dict) -> str:
        """
        Submit an API-format workflow to ComfyUI.
        Returns the prompt_id for tracking.
        """
        payload = {
            "prompt": api_workflow,
            "client_id": self.client_id,
        }
        result = self._post_json("/prompt", payload)
        return result["prompt_id"]

    def get_history(self, prompt_id: str) -> dict | None:
        """Fetch execution history for a given prompt_id."""
        data = self._get(f"/history/{prompt_id}")
        return data.get(prompt_id)

    def wait_for_completion(self, prompt_id: str) -> dict:
        """
        Poll until the prompt finishes executing.
        Returns the history entry with outputs.
        """
        for _ in range(MAX_POLL_ATTEMPTS):
            history = self.get_history(prompt_id)
            if history and history.get("status", {}).get("completed", False):
                return history
            # Also check if status_str_short indicates completion
            if history and "outputs" in history and len(history["outputs"]) > 0:
                return history
            time.sleep(POLL_INTERVAL)
        raise TimeoutError(
            f"Workflow {prompt_id} did not complete within "
            f"{MAX_POLL_ATTEMPTS * POLL_INTERVAL}s"
        )

    def get_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        """Download a generated image from ComfyUI."""
        params = f"?filename={filename}&subfolder={subfolder}&type={folder_type}"
        return self._get_bytes(f"/view{params}")

    def get_output_images(self, history: dict) -> list[dict]:
        """
        Extract image file info from a completed history entry.
        Returns list of {filename, subfolder, type}.
        """
        images = []
        for node_id, node_output in history.get("outputs", {}).items():
            if "images" in node_output:
                for img in node_output["images"]:
                    images.append(img)
        return images

    def list_models(self, model_type: str = "loras") -> list[str]:
        """
        Query available models of a given type.
        model_type: 'loras', 'checkpoints', 'vae', etc.
        """
        try:
            # ComfyUI object_info endpoint has model lists
            data = self._get("/object_info/LoraLoaderModelOnly")
            lora_input = data["LoraLoaderModelOnly"]["input"]["required"]["lora_name"]
            return lora_input[0]  # list of available lora names
        except Exception:
            return []

    def get_queue_status(self) -> dict:
        """Get current queue status."""
        return self._get("/queue")
