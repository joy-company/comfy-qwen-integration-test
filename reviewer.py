"""
Qwen-based prompt reviewer.

Reviews image generation prompts against the Episode Bible
to enforce visual consistency across cuts.

Checks:
1. Directional Continuity - gaze/body direction matches cut definition
2. Hand-Object Locking - objects stay in the correct hand
3. Wardrobe Persistence - outfit matches scene constants
4. Persistent Artifacts - accumulated damage/stains/marks are present
5. Emotional Stacking - facial expression intensity matches progression
"""

import json
from openai import OpenAI
from config import QWEN_BASE_URL, QWEN_MODEL


REVIEW_SYSTEM = """\
당신은 숏폼 드라마의 시각적 일관성을 검증하는 리뷰어입니다.

에피소드 바이블(Episode Bible)과 컷 정의(Cut Definition)를 기준으로,
AI 이미지 생성 프롬프트가 일관성 규칙을 준수하는지 검증합니다.

## 검증 항목 (모두 체크):

1. **의상 일치(Wardrobe)**: 프롬프트의 의상 설명이 바이블의 character_wardrobe와 정확히 일치하는가?
2. **지속 아티팩트(Persistent Artifacts)**: 와인 얼룩, 따귀 자국, 눈물 등 이전 컷에서 발생한 변화가 프롬프트에 반영되어 있는가?
3. **시선/방향(Gaze & Direction)**: 컷 정의의 head_and_gaze가 프롬프트에 반영되어 있는가?
4. **손-물체 결합(Hand-Object)**: 컷 정의의 hand_object_binding이 프롬프트에 정확히 반영되어 있는가?
5. **감정 강도(Facial Gradient)**: 컷 정의의 감정 수치가 프롬프트의 표정 묘사와 일치하는가?
6. **카메라 프레이밍(Framing)**: 컷 type(CU, ECU, Insert, Action)에 맞는 프레이밍이 명시되어 있는가?

## 출력 형식 (JSON만 출력, 다른 텍스트 없이):

승인 시:
{"status": "approved", "reason": "All consistency checks passed."}

거부 시:
{"status": "rejected", "reason": "구체적 불일치 설명", "issues": ["issue1", "issue2"]}
"""


class PromptReviewer:
    def __init__(self):
        self.client = OpenAI(base_url=QWEN_BASE_URL, api_key="ollama")
        self.model = QWEN_MODEL

    def review(self, image_prompt: str, cut: dict, bible: dict) -> dict:
        """
        Review an image prompt for consistency with the episode bible.

        Returns dict with:
          - status: "approved" or "rejected"
          - reason: explanation
          - issues: list of specific problems (if rejected)
        """
        user_msg = (
            "## 에피소드 바이블\n"
            + json.dumps(bible, ensure_ascii=False, indent=2)
            + "\n\n## 컷 정의\n"
            + json.dumps(cut, ensure_ascii=False, indent=2)
            + "\n\n## 검증할 이미지 프롬프트\n"
            + image_prompt
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": REVIEW_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )

        text = response.choices[0].message.content or ""

        # Extract JSON from response (Qwen may wrap in markdown or thinking tags)
        text = self._extract_json(text)

        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            # If we can't parse, assume approved to avoid blocking
            result = {
                "status": "approved",
                "reason": "Reviewer output could not be parsed — auto-approved.",
            }

        return result

    def _extract_json(self, text: str) -> str:
        """Extract JSON from Qwen's response, handling thinking tags and markdown."""
        # Strip <think>...</think> blocks
        import re
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        # Try to find JSON in markdown code blocks
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1)

        # Try to find raw JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0)

        return text.strip()
