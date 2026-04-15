"""
Tool definitions for the Qwen 9B agent.

Each tool is:
1. A JSON schema (OpenAI function-calling format) for the LLM
2. A Python implementation that actually executes the action

Tools:
- load_workflow: Load and parse the ComfyUI workflow JSON
- set_lora: Select and configure a LoRA model in the workflow
- set_prompt: Set positive/negative prompt text
- set_sampler_params: Adjust KSampler parameters (steps, cfg, seed, etc.)
- execute_workflow: Submit the workflow to ComfyUI and wait for results
- list_available_loras: Query ComfyUI for available LoRA models
"""

import json
import random
from pathlib import Path

from comfyui_client import ComfyUIClient
from workflow_converter import (
    load_frontend_workflow,
    convert_to_api_format,
    extract_workflow_info,
)
from config import DEFAULT_WORKFLOW_PATH


# ---------------------------------------------------------------------------
# Shared state: holds the current workflow being manipulated by the agent
# ---------------------------------------------------------------------------

class WorkflowState:
    """Mutable state shared across tool calls within a single agent run."""

    def __init__(self):
        self.frontend_workflow: dict | None = None
        self.api_workflow: dict | None = None
        self.workflow_info: dict | None = None
        self.comfyui = ComfyUIClient()
        self.last_prompt_id: str | None = None
        self.last_images: list[dict] = []

    def ensure_loaded(self) -> bool:
        return self.api_workflow is not None


_state = WorkflowState()


def get_state() -> WorkflowState:
    return _state


def reset_state():
    global _state
    _state = WorkflowState()


# ---------------------------------------------------------------------------
# Tool: load_workflow
# ---------------------------------------------------------------------------

LOAD_WORKFLOW_SCHEMA = {
    "type": "function",
    "function": {
        "name": "load_workflow",
        "description": (
            "ComfyUI 워크플로우 JSON 파일을 로드합니다. "
            "워크플로우의 구조(노드, LoRA, 프롬프트, 샘플러 설정 등)를 분석하여 반환합니다. "
            "다른 도구를 사용하기 전에 반드시 이 도구를 먼저 호출해야 합니다."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "workflow_path": {
                    "type": "string",
                    "description": "워크플로우 JSON 파일 경로 (기본값: flux_depth_lora_example.json)",
                }
            },
            "required": [],
        },
    },
}


def load_workflow(workflow_path: str = None) -> str:
    state = get_state()
    path = Path(workflow_path) if workflow_path else Path(DEFAULT_WORKFLOW_PATH)

    if not path.exists():
        return json.dumps({"error": f"파일을 찾을 수 없습니다: {path}"}, ensure_ascii=False)

    state.frontend_workflow = load_frontend_workflow(path)
    state.api_workflow = convert_to_api_format(state.frontend_workflow)
    state.workflow_info = extract_workflow_info(state.frontend_workflow)

    return json.dumps({
        "status": "success",
        "message": "워크플로우가 성공적으로 로드되었습니다.",
        "workflow_info": {
            "lora": state.workflow_info["lora"],
            "base_model": state.workflow_info["base_model"],
            "positive_prompt": state.workflow_info["positive_prompt"],
            "negative_prompt": state.workflow_info["negative_prompt"],
            "sampler": state.workflow_info["sampler"],
            "input_image": state.workflow_info["input_image"],
            "total_nodes": len(state.api_workflow),
        },
    }, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Tool: set_lora
# ---------------------------------------------------------------------------

SET_LORA_SCHEMA = {
    "type": "function",
    "function": {
        "name": "set_lora",
        "description": (
            "워크플로우에서 사용할 LoRA 모델을 선택하고 강도(strength)를 설정합니다. "
            "워크플로우가 먼저 로드되어 있어야 합니다."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "lora_name": {
                    "type": "string",
                    "description": "사용할 LoRA 모델 파일명 (예: flux1-depth-dev-lora.safetensors)",
                },
                "strength": {
                    "type": "number",
                    "description": "LoRA 적용 강도 (0.0 ~ 2.0, 기본값 1.0)",
                    "default": 1.0,
                },
            },
            "required": ["lora_name"],
        },
    },
}


def set_lora(lora_name: str, strength: float = 1.0) -> str:
    state = get_state()
    if not state.ensure_loaded():
        return json.dumps({"error": "워크플로우가 로드되지 않았습니다. load_workflow를 먼저 호출하세요."}, ensure_ascii=False)

    lora_info = state.workflow_info.get("lora")
    if not lora_info:
        return json.dumps({"error": "워크플로우에 LoRA 노드가 없습니다."}, ensure_ascii=False)

    node_id = lora_info["node_id"]
    if node_id in state.api_workflow:
        state.api_workflow[node_id]["inputs"]["lora_name"] = lora_name
        state.api_workflow[node_id]["inputs"]["strength_model"] = strength
        lora_info["lora_name"] = lora_name
        lora_info["strength"] = strength

    return json.dumps({
        "status": "success",
        "message": f"LoRA가 설정되었습니다: {lora_name} (강도: {strength})",
        "node_id": node_id,
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool: set_prompt
# ---------------------------------------------------------------------------

SET_PROMPT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "set_prompt",
        "description": (
            "이미지 생성에 사용할 프롬프트(positive/negative)를 설정합니다. "
            "프롬프트를 검증하고 필요 시 수정한 후 워크플로우에 반영합니다."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "positive_prompt": {
                    "type": "string",
                    "description": "생성할 이미지를 설명하는 긍정 프롬프트",
                },
                "negative_prompt": {
                    "type": "string",
                    "description": "피하고 싶은 요소를 설명하는 부정 프롬프트 (선택사항)",
                    "default": "",
                },
            },
            "required": ["positive_prompt"],
        },
    },
}


def set_prompt(positive_prompt: str, negative_prompt: str = "") -> str:
    state = get_state()
    if not state.ensure_loaded():
        return json.dumps({"error": "워크플로우가 로드되지 않았습니다. load_workflow를 먼저 호출하세요."}, ensure_ascii=False)

    results = {}

    # Set positive prompt
    pos_info = state.workflow_info.get("positive_prompt")
    if pos_info:
        node_id = pos_info["node_id"]
        if node_id in state.api_workflow:
            state.api_workflow[node_id]["inputs"]["text"] = positive_prompt
            pos_info["text"] = positive_prompt
            results["positive"] = {"node_id": node_id, "text": positive_prompt}

    # Set negative prompt
    neg_info = state.workflow_info.get("negative_prompt")
    if neg_info and negative_prompt:
        node_id = neg_info["node_id"]
        if node_id in state.api_workflow:
            state.api_workflow[node_id]["inputs"]["text"] = negative_prompt
            neg_info["text"] = negative_prompt
            results["negative"] = {"node_id": node_id, "text": negative_prompt}

    return json.dumps({
        "status": "success",
        "message": "프롬프트가 설정되었습니다.",
        "prompts": results,
    }, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Tool: set_sampler_params
# ---------------------------------------------------------------------------

SET_SAMPLER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "set_sampler_params",
        "description": (
            "KSampler의 파라미터를 설정합니다: seed, steps, cfg, sampler_name, scheduler, denoise. "
            "지정하지 않은 파라미터는 기존 값을 유지합니다."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "seed": {
                    "type": "integer",
                    "description": "랜덤 시드 값. -1이면 랜덤 생성.",
                },
                "steps": {
                    "type": "integer",
                    "description": "샘플링 스텝 수 (기본값: 20)",
                },
                "cfg": {
                    "type": "number",
                    "description": "CFG 스케일 (기본값: 1.0)",
                },
                "sampler_name": {
                    "type": "string",
                    "description": "샘플러 이름 (euler, dpmpp_2m 등)",
                },
                "scheduler": {
                    "type": "string",
                    "description": "스케줄러 이름 (normal, karras 등)",
                },
                "denoise": {
                    "type": "number",
                    "description": "디노이즈 강도 (0.0 ~ 1.0)",
                },
            },
            "required": [],
        },
    },
}


def set_sampler_params(**kwargs) -> str:
    state = get_state()
    if not state.ensure_loaded():
        return json.dumps({"error": "워크플로우가 로드되지 않았습니다."}, ensure_ascii=False)

    sampler_info = state.workflow_info.get("sampler")
    if not sampler_info:
        return json.dumps({"error": "워크플로우에 KSampler 노드가 없습니다."}, ensure_ascii=False)

    node_id = sampler_info["node_id"]
    node = state.api_workflow.get(node_id)
    if not node:
        return json.dumps({"error": f"노드 {node_id}를 찾을 수 없습니다."}, ensure_ascii=False)

    updated = {}
    param_map = {
        "seed": "seed",
        "steps": "steps",
        "cfg": "cfg",
        "sampler_name": "sampler_name",
        "scheduler": "scheduler",
        "denoise": "denoise",
    }

    for param, api_key in param_map.items():
        if param in kwargs and kwargs[param] is not None:
            value = kwargs[param]
            if param == "seed" and value == -1:
                value = random.randint(0, 2**53)
            node["inputs"][api_key] = value
            sampler_info[param] = value
            updated[param] = value

    return json.dumps({
        "status": "success",
        "message": "샘플러 파라미터가 업데이트되었습니다.",
        "updated": updated,
        "current_settings": {
            k: sampler_info.get(k) for k in param_map
        },
    }, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Tool: execute_workflow
# ---------------------------------------------------------------------------

EXECUTE_WORKFLOW_SCHEMA = {
    "type": "function",
    "function": {
        "name": "execute_workflow",
        "description": (
            "현재 설정된 워크플로우를 ComfyUI 서버에 제출하여 이미지 생성을 시작합니다. "
            "생성이 완료될 때까지 대기한 후 결과를 반환합니다."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}


OUTPUT_DIR = Path(__file__).parent / "output"


def execute_workflow(**kwargs) -> str:
    state = get_state()
    if not state.ensure_loaded():
        return json.dumps({"error": "워크플로우가 로드되지 않았습니다."}, ensure_ascii=False)

    # Check ComfyUI is alive
    if not state.comfyui.is_alive():
        return json.dumps({
            "error": "ComfyUI 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.",
        }, ensure_ascii=False)

    # Print the API workflow being submitted
    print("\n" + "=" * 60)
    print("[Workflow] ComfyUI에 제출할 API 워크플로우:")
    print("=" * 60)
    for nid, node in sorted(state.api_workflow.items(), key=lambda x: int(x[0])):
        inputs_str = ", ".join(
            f"{k}={json.dumps(v, ensure_ascii=False) if isinstance(v, str) else v}"
            for k, v in node["inputs"].items()
        )
        print(f"  [{nid}] {node['class_type']}: {inputs_str}")
    print("=" * 60 + "\n")

    # Submit workflow
    try:
        prompt_id = state.comfyui.queue_prompt(state.api_workflow)
        state.last_prompt_id = prompt_id
    except Exception as e:
        return json.dumps({
            "error": f"워크플로우 제출 실패: {str(e)}",
        }, ensure_ascii=False)

    # Wait for completion
    try:
        history = state.comfyui.wait_for_completion(prompt_id)
    except TimeoutError as e:
        return json.dumps({
            "error": str(e),
            "prompt_id": prompt_id,
        }, ensure_ascii=False)

    # Extract output images
    images = state.comfyui.get_output_images(history)
    state.last_images = images

    # Always save images to local output folder
    save_dir = OUTPUT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for img_info in images:
        img_data = state.comfyui.get_image(
            img_info["filename"],
            img_info.get("subfolder", ""),
            img_info.get("type", "output"),
        )
        out_path = save_dir / img_info["filename"]
        out_path.write_bytes(img_data)
        saved_paths.append(str(out_path))

    return json.dumps({
        "status": "success",
        "message": "이미지 생성이 완료되었습니다!",
        "prompt_id": prompt_id,
        "images": [
            {
                "filename": img["filename"],
                "subfolder": img.get("subfolder", ""),
                "type": img.get("type", "output"),
            }
            for img in images
        ],
        "saved_to": saved_paths,
    }, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Tool: list_available_loras
# ---------------------------------------------------------------------------

LIST_LORAS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_available_loras",
        "description": "ComfyUI 서버에서 사용 가능한 LoRA 모델 목록을 조회합니다.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}


def list_available_loras() -> str:
    state = get_state()
    if not state.comfyui.is_alive():
        return json.dumps({
            "error": "ComfyUI 서버에 연결할 수 없습니다.",
        }, ensure_ascii=False)

    loras = state.comfyui.list_models("loras")
    return json.dumps({
        "status": "success",
        "available_loras": loras,
        "count": len(loras),
    }, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Registry: all tools available to the agent
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    LOAD_WORKFLOW_SCHEMA,
    SET_LORA_SCHEMA,
    SET_PROMPT_SCHEMA,
    SET_SAMPLER_SCHEMA,
    EXECUTE_WORKFLOW_SCHEMA,
    LIST_LORAS_SCHEMA,
]

TOOL_FUNCTIONS = {
    "load_workflow": load_workflow,
    "set_lora": set_lora,
    "set_prompt": set_prompt,
    "set_sampler_params": set_sampler_params,
    "execute_workflow": execute_workflow,
    "list_available_loras": list_available_loras,
}


def execute_tool(name: str, arguments: dict) -> str:
    """Dispatch a tool call by name with given arguments."""
    func = TOOL_FUNCTIONS.get(name)
    if not func:
        return json.dumps({"error": f"알 수 없는 도구: {name}"}, ensure_ascii=False)
    try:
        return func(**arguments)
    except Exception as e:
        return json.dumps({"error": f"도구 실행 오류 ({name}): {str(e)}"}, ensure_ascii=False)
