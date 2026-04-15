"""
Convert ComfyUI frontend workflow JSON to API-format prompt.

The frontend format stores visual graph data (positions, links, groups, etc.).
The API format is a flat dict keyed by node ID, each with class_type + inputs.

This converter:
1. Parses the frontend JSON
2. Resolves widget values and link connections
3. Produces a dict suitable for POST /prompt
"""

import json
from pathlib import Path


def load_frontend_workflow(path: str | Path) -> dict:
    """Load a ComfyUI frontend workflow JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_link_map(workflow: dict) -> dict:
    """
    Build a lookup: link_id -> (origin_node_id, origin_slot_index).
    Links live at workflow["links"] (flat format) or may be embedded in
    subgraphs. We handle both.
    """
    link_map = {}

    # Top-level links (standard ComfyUI format)
    for link in workflow.get("links", []):
        # link format: [link_id, origin_node, origin_slot, target_node, target_slot, type]
        if isinstance(link, list) and len(link) >= 5:
            link_id, origin_node, origin_slot = link[0], link[1], link[2]
            link_map[link_id] = (origin_node, origin_slot)

    return link_map


def _get_widget_names_for_type(class_type: str) -> list[str]:
    """
    Return the expected widget parameter names for known node types.
    This maps class_type to the ordered list of widget value names.
    """
    # Use None for frontend-only widgets that should not be sent to the API
    widget_map = {
        "KSampler": ["seed", None, "steps", "cfg", "sampler_name", "scheduler", "denoise"],
        "KSamplerSelect": ["sampler_name"],
        "CLIPTextEncode": ["text"],
        "SaveImage": ["filename_prefix"],
        "LoadImage": ["image", None],
        "UNETLoader": ["unet_name", "weight_dtype"],
        "LoraLoaderModelOnly": ["lora_name", "strength_model"],
        "VAELoader": ["vae_name"],
        "DualCLIPLoader": ["clip_name1", "clip_name2", "type"],
        "EmptyLatentImage": ["width", "height", "batch_size"],
        "BasicScheduler": ["scheduler", "steps", "denoise"],
        "RandomNoise": ["noise_seed"],
        "SamplerCustomAdvanced": [],
        "BasicGuider": [],
        "FluxGuidance": ["guidance"],
        "VAEDecode": [],
        "VAEEncode": [],
        "ImageResize+": ["width", "height", "interpolation", "method", "condition", "multiple_of"],
        "MarkdownNote": ["text"],
        "InstructPixToPixConditioning": [],
        "CheckpointLoaderSimple": ["ckpt_name"],
    }
    return widget_map.get(class_type, [])


def convert_to_api_format(workflow: dict) -> dict:
    """
    Convert a frontend workflow to ComfyUI API format.

    Returns a dict like:
    {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 229472716717627,
                "steps": 20,
                ...
                "model": ["4", 0],   # link to node 4, output slot 0
            }
        },
        ...
    }
    """
    nodes = workflow.get("nodes", [])
    link_map = _build_link_map(workflow)

    api_prompt = {}

    for node in nodes:
        node_id = str(node["id"])
        class_type = node.get("type", "")

        # Skip non-executable node types (notes, reroutes, etc.)
        if class_type in ("MarkdownNote", "Reroute", "Note", "PrimitiveNode"):
            continue

        inputs = {}

        # 1) Map widget values to named parameters
        widget_values = node.get("widgets_values", [])
        widget_names = _get_widget_names_for_type(class_type)

        for i, name in enumerate(widget_names):
            if i < len(widget_values) and name is not None:
                inputs[name] = widget_values[i]

        # For unknown node types, store widgets by index as fallback
        if not widget_names and widget_values:
            # Try to use input definitions to guess names
            input_defs = [inp for inp in node.get("inputs", []) if inp.get("widget")]
            if not input_defs:
                for i, val in enumerate(widget_values):
                    inputs[f"widget_{i}"] = val

        # 2) Map linked inputs: input slots connected via links
        for inp in node.get("inputs", []):
            link_id = inp.get("link")
            if link_id is not None and link_id in link_map:
                origin_node, origin_slot = link_map[link_id]
                inputs[inp["name"]] = [str(origin_node), origin_slot]

        api_prompt[node_id] = {
            "class_type": class_type,
            "inputs": inputs,
        }

    return api_prompt


def extract_workflow_info(workflow: dict) -> dict:
    """
    Extract human-readable summary of key nodes for the agent.
    """
    nodes = workflow.get("nodes", [])
    info = {
        "lora": None,
        "base_model": None,
        "positive_prompt": None,
        "negative_prompt": None,
        "sampler": None,
        "input_image": None,
        "all_node_ids": {},
    }

    # Build a map: origin_node_id -> [(target_node_id, target_slot)]
    # Used to determine which CLIPTextEncode is positive vs negative
    link_targets = {}
    for link in workflow.get("links", []):
        if isinstance(link, list) and len(link) >= 5:
            origin_node = link[1]
            target_node = link[3]
            target_slot = link[4]
            link_targets.setdefault(origin_node, []).append((target_node, target_slot))

    # Find KSampler node id for link-based prompt detection
    ksampler_id = None
    for node in nodes:
        if node.get("type") == "KSampler":
            ksampler_id = node["id"]
            break

    # Collect unresolved CLIPTextEncode nodes for link-based fallback
    clip_text_nodes = []

    for node in nodes:
        class_type = node.get("type", "")
        node_id = str(node["id"])
        title = node.get("title", class_type)
        widgets = node.get("widgets_values", [])

        info["all_node_ids"][node_id] = {"class_type": class_type, "title": title}

        if class_type == "LoraLoaderModelOnly" and widgets:
            info["lora"] = {
                "node_id": node_id,
                "lora_name": widgets[0] if len(widgets) > 0 else None,
                "strength": widgets[1] if len(widgets) > 1 else 1.0,
            }
        elif class_type in ("UNETLoader", "CheckpointLoaderSimple") and widgets:
            info["base_model"] = {
                "node_id": node_id,
                "model_name": widgets[0] if len(widgets) > 0 else None,
            }
        elif class_type == "CLIPTextEncode":
            if "Positive" in title:
                info["positive_prompt"] = {
                    "node_id": node_id,
                    "text": widgets[0] if widgets else "",
                }
            elif "Negative" in title:
                info["negative_prompt"] = {
                    "node_id": node_id,
                    "text": widgets[0] if widgets else "",
                }
            else:
                clip_text_nodes.append((node_id, widgets))
        elif class_type == "KSampler" and widgets:
            info["sampler"] = {
                "node_id": node_id,
                "seed": widgets[0] if len(widgets) > 0 else None,
                "steps": widgets[2] if len(widgets) > 2 else 20,
                "cfg": widgets[3] if len(widgets) > 3 else 1,
                "sampler_name": widgets[4] if len(widgets) > 4 else "euler",
                "scheduler": widgets[5] if len(widgets) > 5 else "normal",
                "denoise": widgets[6] if len(widgets) > 6 else 1.0,
            }
        elif class_type == "LoadImage" and widgets:
            info["input_image"] = {
                "node_id": node_id,
                "image": widgets[0] if widgets else None,
            }

    # Resolve untitled CLIPTextEncode nodes by tracing links to KSampler
    # KSampler inputs: slot 1 = positive, slot 2 = negative
    for node_id, widgets in clip_text_nodes:
        resolved = False
        if ksampler_id is not None:
            for target_node, target_slot in link_targets.get(int(node_id), []):
                if target_node == ksampler_id:
                    entry = {
                        "node_id": node_id,
                        "text": widgets[0] if widgets else "",
                    }
                    if target_slot == 1 and info["positive_prompt"] is None:
                        info["positive_prompt"] = entry
                        resolved = True
                    elif target_slot == 2 and info["negative_prompt"] is None:
                        info["negative_prompt"] = entry
                        resolved = True
                    break
        # Fallback: assign to first empty slot
        if not resolved:
            entry = {"node_id": node_id, "text": widgets[0] if widgets else ""}
            if info["positive_prompt"] is None:
                info["positive_prompt"] = entry
            elif info["negative_prompt"] is None:
                info["negative_prompt"] = entry

    return info
