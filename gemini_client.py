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
- 70%+ of cuts must be Close-Up (CU) or Extreme Close-Up (ECU).
- Every spoken line MUST produce a Reaction Shot cut for the listener.
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
# API calls
# ---------------------------------------------------------------------------

def generate_bible(script: str) -> dict:
    """Generate an Episode Bible JSON from the script."""
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


def divide_into_cuts(script: str, bible: dict) -> list[dict]:
    """Divide the script into production cuts."""
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


def generate_image_prompt(cut: dict, bible: dict) -> str:
    """Generate an image generation prompt for a single cut."""
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


def revise_image_prompt(
    original_prompt: str,
    cut: dict,
    bible: dict,
    feedback: str,
) -> str:
    """Revise an image prompt based on reviewer feedback."""
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
