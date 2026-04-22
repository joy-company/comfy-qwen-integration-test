"""
Gemini API client for episode pipeline.

Handles:
1. Episode Bible generation (Character_DNA, Scene_Constants)
2. Script → Cut division
3. Cut → Image prompt generation
4. Prompt revision based on reviewer feedback
"""

import json
import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL


genai.configure(api_key=GEMINI_API_KEY)
_model = genai.GenerativeModel(GEMINI_MODEL)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

BIBLE_SYSTEM = """\
You are a visual continuity supervisor for a short-form vertical (9:16) drama.

Given an episode script, produce an Episode Bible JSON with this exact structure:

{
  "episode_id": "EP01",
  "characters": {
    "<NAME>": {
      "character_DNA": {
        "face_taxonomy": "bone structure, eye shape/angle, nose height, lip thickness, distinguishing marks",
        "body_proportions": "height, build, shoulder width, default posture",
        "persona_markers": "core deficiency, habitual expression set"
      },
      "default_wardrobe": "outfit worn at start of episode"
    }
  },
  "scenes": {
    "<SCENE_ID>": {
      "location": "physical space description",
      "lighting_environment": "key light direction, color temperature, shadow intensity",
      "spatial_layout_916": "vertical frame composition notes",
      "character_wardrobe": {
        "<NAME>": "exact outfit in this scene, texture, color, design details"
      },
      "persistent_artifacts": [
        "changes that carry forward from prior action, e.g. slap mark, wine stain"
      ]
    }
  }
}

Rules:
- Extract EVERY character mentioned, even minor ones.
- For wardrobe, be extremely specific: fabric, color, fit, visible details.
- For persistent_artifacts, track cumulative damage/changes across the script.
- Output ONLY valid JSON, no markdown fences, no commentary.
"""

CUT_DIVISION_SYSTEM = """\
You are a short-form drama editor dividing a script into production cuts.

Follow the 3-7-21 framework:
- 0~3s: Sensory Hook (insert shots, slaps, screams)
- 3~7s: Theme Anchoring (key declaration dialogue)
- 7~21s: Engagement Escalation (humiliation, crisis)
- 21~90s: Spiral Escalation + Cliffhanger

Rules:
- TARGET 25-27 CUTS TOTAL per episode. Do NOT exceed 30 cuts.
- Combine consecutive reaction shots into the prior cut when possible.
- Only create a separate Reaction Shot cut for the most emotionally impactful moments.
- 70%+ of cuts must be Close-Up (CU) or Extreme Close-Up (ECU).
- Preserve 100% of original dialogue — never edit, omit, or add lines.
- Use Insert shots for props with information value (documents, phones, etc.).
- Cut duration: mostly 2-3 seconds, 1s for emotional peaks.

Output a JSON array with this structure for each cut:

[
  {
    "cut_number": 1,
    "scene_id": "S#01",
    "type": "Insert | CU | ECU | Action | Reaction",
    "duration_sec": 1.5,
    "action": "Visual description of what is shown",
    "dialogue": {
      "speaker": "Name or null",
      "line": "Exact original line or null"
    },
    "head_and_gaze": {
      "character": "Name",
      "tilt": "direction and degrees",
      "target": "what they look at"
    },
    "hand_object_binding": "what is in which hand, or null",
    "facial_gradient": "emotion and intensity 0-100",
    "depth_stacking": "foreground / midground / background content"
  }
]

Output ONLY the JSON array — no markdown, no commentary.
"""

PROMPT_GEN_SYSTEM = """\
You are an image prompt engineer for AI image generation (Stable Diffusion / FLUX).

Given a cut definition and the episode bible, write a single detailed image generation
prompt in English that will produce the exact frame described.

Rules:
- ALL human characters are Korean. Always specify "Korean man", "Korean woman", etc.
- Start with the subject and their exact appearance from character_DNA.
- Include EXACT wardrobe from the scene's character_wardrobe field.
- Include any persistent_artifacts (wine stains, slap marks, tears, etc.).
- Describe the lighting from lighting_environment.
- Specify camera framing: CU = face and upper shoulders, ECU = eyes/mouth only,
  Action = waist up or full body, Insert = object close-up.
- Include the facial expression matching facial_gradient.
- Include gaze direction from head_and_gaze.
- Include hand positions and held objects from hand_object_binding.
- End with quality tags: photorealistic, cinematic, 9:16 vertical, shallow depth of field.
- Do NOT include any dialogue text in the prompt.
- Output ONLY the prompt text — one paragraph, no line breaks, no commentary.
"""

DRAMA_PROMPT_GEN_SYSTEM = """\
You are an image prompt engineer for AI image generation (Stable Diffusion / FLUX).

You will receive a drama cut analysis (from a multi-agent pipeline) and the episode bible.
The drama cut contains rich directorial and cinematographic information. Your job is to
merge it with the bible's character consistency data into a single image generation prompt.

Rules:
- ALL human characters are Korean. Always specify "Korean man", "Korean woman", etc.
- Start with the subject and their exact appearance from character_DNA in the bible.
- Include EXACT wardrobe from the scene's character_wardrobe field in the bible.
- Include any persistent_artifacts (wine stains, slap marks, tears, etc.) from the bible.
- Use the drama cut's "description" as the primary scene content.
- Use the drama cut's "technical_specs" for lens and aperture details.
- Use the drama cut's "lighting" for light direction, style, and color mood.
- Use the drama cut's "composition" for shot type, focal point, and camera movement.
- Use the drama cut's "emotional_intent" to guide the facial expression and mood.
- End with quality tags: photorealistic, cinematic, 9:16 vertical, shallow depth of field.
- Do NOT include any dialogue text in the prompt.
- Output ONLY the prompt text — one paragraph, no line breaks, no commentary.
"""

PROMPT_REVISE_SYSTEM = """\
You are an image prompt engineer revising a prompt based on reviewer feedback.

You will receive:
1. The original image prompt
2. The episode bible (for reference)
3. The cut definition
4. Reviewer feedback explaining what is inconsistent

Revise the prompt to fix the issues identified by the reviewer.
Preserve everything else that was correct.
Output ONLY the revised prompt text — one paragraph, no commentary.
"""


# ---------------------------------------------------------------------------
# Retry wrapper — handles 503/429 from Gemini API
# ---------------------------------------------------------------------------

import time
import threading
import sys

MAX_RETRIES = 4
RETRY_DELAYS = [5, 10, 20, 40]  # seconds — exponential backoff


def _spinner(label, stop_event):
    """Background spinner that prints a dot every 3 seconds while waiting."""
    start = time.time()
    while not stop_event.is_set():
        elapsed = int(time.time() - start)
        sys.stdout.write(f"\r  ⏳ {label}... {elapsed}s ")
        sys.stdout.flush()
        stop_event.wait(3)
    # Clear the spinner line
    sys.stdout.write("\r" + " " * 60 + "\r")
    sys.stdout.flush()


def _call_with_retry(fn, *args, label="Gemini API", **kwargs):
    """Call a Gemini API function with retry on transient errors (503, 429)."""
    for attempt in range(MAX_RETRIES):
        stop_event = threading.Event()
        spinner_thread = threading.Thread(target=_spinner, args=(label, stop_event), daemon=True)
        spinner_thread.start()
        try:
            result = fn(*args, **kwargs)
            stop_event.set()
            spinner_thread.join()
            return result
        except Exception as e:
            stop_event.set()
            spinner_thread.join()
            err_str = str(e)
            is_transient = "503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str
            if is_transient and attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAYS[attempt]
                print(f"  ⟳ Gemini {err_str[:80]}... 재시도 {attempt + 1}/{MAX_RETRIES} ({delay}s 대기)")
                time.sleep(delay)
                continue
            raise


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def generate_bible(script: str) -> dict:
    """Generate an Episode Bible JSON from the script. Falls back to Ollama if Gemini fails."""
    def _call():
        response = _model.generate_content(
            [
                {"role": "user", "parts": [BIBLE_SYSTEM + "\n\nHere is the episode script:\n\n" + script]},
            ],
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                response_mime_type="application/json",
            ),
        )
        return json.loads(response.text)

    try:
        return _call_with_retry(_call, label="바이블 생성 (Gemini)")
    except Exception as e:
        print(f"  ⚠ Gemini 바이블 생성 실패: {e}")
        print("  ⟳ Ollama (Gemma 4) 폴백으로 바이블 생성 시도 중...")
        return _generate_bible_ollama(script)


def _generate_bible_ollama(script: str) -> dict:
    """Fallback: generate bible via Ollama (Gemma 4) when Gemini is unavailable."""
    import re
    from config import QWEN_BASE_URL

    # Use Ollama's chat endpoint directly
    ollama_url = "http://127.0.0.1:11434/api/chat"
    ollama_model = "gemma4:26b"

    prompt = BIBLE_SYSTEM + "\n\nHere is the episode script:\n\n" + script

    body = {
        "model": ollama_model,
        "stream": False,
        "options": {"temperature": 0.3},
        "messages": [
            {"role": "system", "content": "You must respond with ONLY valid JSON. No markdown fences, no commentary."},
            {"role": "user", "content": prompt},
        ],
    }

    import urllib.request
    stop_event = threading.Event()
    spinner_thread = threading.Thread(target=_spinner, args=("바이블 생성 (Ollama 폴백)", stop_event), daemon=True)
    spinner_thread.start()

    try:
        req = urllib.request.Request(
            ollama_url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        stop_event.set()
        spinner_thread.join()

        content = data.get("message", {}).get("content", "")
        # Strip markdown fences and thinking tokens if present
        content = re.sub(r"<\|think\|>[\s\S]*?<\|/think\|>", "", content)
        content = re.sub(r"```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```", "", content)
        content = content.strip()

        return json.loads(content)
    except Exception:
        stop_event.set()
        spinner_thread.join()
        raise


def divide_into_cuts(script: str, bible: dict) -> list[dict]:
    """Divide the script into production cuts."""
    def _call():
        prompt = (
            CUT_DIVISION_SYSTEM
            + "\n\n--- EPISODE BIBLE ---\n"
            + json.dumps(bible, ensure_ascii=False, indent=2)
            + "\n\n--- EPISODE SCRIPT ---\n"
            + script
        )
        response = _model.generate_content(
            [{"role": "user", "parts": [prompt]}],
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                response_mime_type="application/json",
            ),
        )
        return json.loads(response.text)
    return _call_with_retry(_call, label="컷 분할")


def generate_image_prompt(cut: dict, bible: dict) -> str:
    """Generate an image generation prompt for a single cut."""
    cut_num = cut.get("cut_number", "?")
    def _call():
        prompt = (
            PROMPT_GEN_SYSTEM
            + "\n\n--- EPISODE BIBLE ---\n"
            + json.dumps(bible, ensure_ascii=False, indent=2)
            + "\n\n--- CUT DEFINITION ---\n"
            + json.dumps(cut, ensure_ascii=False, indent=2)
        )
        response = _model.generate_content(
            [{"role": "user", "parts": [prompt]}],
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
            ),
        )
        return response.text.strip()
    return _call_with_retry(_call, label=f"이미지 프롬프트 (컷 {cut_num})")


def generate_image_prompt_from_drama_cut(drama_cut: dict, bible: dict) -> str:
    """Generate an image prompt from a drama pipeline cut + episode bible."""
    cut_num = drama_cut.get("cut_number", "?")
    def _call():
        prompt = (
            DRAMA_PROMPT_GEN_SYSTEM
            + "\n\n--- EPISODE BIBLE ---\n"
            + json.dumps(bible, ensure_ascii=False, indent=2)
            + "\n\n--- DRAMA CUT ANALYSIS ---\n"
            + json.dumps(drama_cut, ensure_ascii=False, indent=2)
        )
        response = _model.generate_content(
            [{"role": "user", "parts": [prompt]}],
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
            ),
        )
        return response.text.strip()
    return _call_with_retry(_call, label=f"드라마 컷 프롬프트 (컷 {cut_num})")


def revise_image_prompt(
    original_prompt: str,
    cut: dict,
    bible: dict,
    feedback: str,
) -> str:
    """Revise an image prompt based on reviewer feedback."""
    def _call():
        prompt = (
            PROMPT_REVISE_SYSTEM
            + "\n\n--- EPISODE BIBLE ---\n"
            + json.dumps(bible, ensure_ascii=False, indent=2)
            + "\n\n--- CUT DEFINITION ---\n"
            + json.dumps(cut, ensure_ascii=False, indent=2)
            + "\n\n--- ORIGINAL PROMPT ---\n"
            + original_prompt
            + "\n\n--- REVIEWER FEEDBACK ---\n"
            + feedback
        )
        response = _model.generate_content(
            [{"role": "user", "parts": [prompt]}],
            generation_config=genai.types.GenerationConfig(
                temperature=0.5,
            ),
        )
        return response.text.strip()
    cut_num = cut.get("cut_number", "?")
    return _call_with_retry(_call, label=f"프롬프트 수정 (컷 {cut_num})")
