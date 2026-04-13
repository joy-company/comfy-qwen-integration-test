"""
Qwen 9B Agent with tool-calling capability.

Uses Ollama's OpenAI-compatible API to run Qwen 9B as an agent that can:
1. Validate and refine image generation prompts
2. Load ComfyUI workflows
3. Select appropriate LoRA models
4. Configure generation parameters
5. Execute workflows to generate images

The agent runs a ReAct-style loop:
  Think -> Tool Call -> Observe -> Think -> ... -> Final Answer
"""

import json
from openai import OpenAI

from config import QWEN_BASE_URL, QWEN_MODEL
from tools import TOOL_SCHEMAS, execute_tool


# ---------------------------------------------------------------------------
# System prompt for the Qwen 9B agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
당신은 ComfyUI 이미지 생성 파이프라인을 제어하는 전문 에이전트입니다.

## 역할
- 사용자로부터 이미지 생성 요청을 받아 프롬프트를 검증하고 최적화합니다
- ComfyUI 워크플로우를 로드하고 적절한 LoRA 모델과 파라미터를 설정합니다
- 워크플로우를 실행하여 이미지를 생성합니다

## 프롬프트 검증 체크리스트
프롬프트를 설정하기 전에 다음을 확인하세요:
1. 프롬프트가 영어로 작성되어 있는지 (한국어인 경우 영어로 번역)
2. 주요 피사체/장면이 명확하게 설명되어 있는지
3. 스타일, 조명, 분위기 등 시각적 세부사항이 포함되어 있는지
4. 부정적 요소(negative prompt)가 적절히 지정되어 있는지
5. 프롬프트 길이가 적절한지 (너무 짧거나 너무 길지 않은지)

## 작업 순서
반드시 다음 순서로 작업하세요:
1. load_workflow: 워크플로우 파일을 먼저 로드
2. 워크플로우 정보를 확인하고 LoRA 모델이 적절한지 판단
3. set_lora: 필요 시 LoRA 모델 변경 (워크플로우에 이미 정의된 LoRA가 적절하면 그대로 사용)
4. set_prompt: 검증/수정된 프롬프트 설정
5. set_sampler_params: 필요 시 샘플러 파라미터 조정
6. execute_workflow: 워크플로우 실행

## 주의사항
- 도구 호출 결과를 반드시 확인하고, 오류가 있으면 적절히 대처하세요
- 워크플로우에 이미 정의된 LoRA 모델 정보를 존중하세요
- 사용자가 명시적으로 변경을 요청하지 않은 파라미터는 기본값을 유지하세요
- 최종 결과를 사용자에게 명확하게 보고하세요
"""


class QwenAgent:
    """
    Qwen 9B agent that uses tool calling to control ComfyUI.
    """

    def __init__(
        self,
        base_url: str = QWEN_BASE_URL,
        model: str = QWEN_MODEL,
        max_iterations: int = 15,
    ):
        self.client = OpenAI(base_url=base_url, api_key="ollama")
        self.model = model
        self.max_iterations = max_iterations
        self.messages: list[dict] = []

    def _init_messages(self, user_request: str):
        """Initialize the conversation with system prompt and user request."""
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_request},
        ]

    def _call_llm(self) -> dict:
        """Make a single LLM call with tool definitions."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.7,
        )
        return response.choices[0].message

    def _process_tool_calls(self, message) -> bool:
        """
        Process any tool calls in the message.
        Returns True if tool calls were made, False if the agent is done.
        """
        if not message.tool_calls:
            return False

        # Append the assistant message (with tool calls) to history
        self.messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ],
        })

        # Execute each tool and add results
        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                arguments = {}

            print(f"  [Tool] {func_name}({json.dumps(arguments, ensure_ascii=False)[:200]})")

            result = execute_tool(func_name, arguments)
            print(f"  [Result] {result[:300]}...")

            self.messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

        return True

    def run(self, user_request: str) -> str:
        """
        Run the agent loop for a given user request.

        Args:
            user_request: Natural language request for image generation.

        Returns:
            The agent's final text response summarizing what was done.
        """
        self._init_messages(user_request)
        print(f"\n{'='*60}")
        print(f"[Agent] 요청 수신: {user_request[:100]}...")
        print(f"{'='*60}")

        for iteration in range(self.max_iterations):
            print(f"\n--- Iteration {iteration + 1} ---")

            message = self._call_llm()

            # If there are tool calls, process them and continue
            if message.tool_calls:
                self._process_tool_calls(message)
                if message.content:
                    print(f"  [Think] {message.content[:200]}")
                continue

            # No tool calls = agent is done, return final response
            final_response = message.content or "작업이 완료되었습니다."
            print(f"\n[Agent] 최종 응답:\n{final_response}")
            return final_response

        return "최대 반복 횟수에 도달했습니다. 작업을 완료하지 못했을 수 있습니다."

    def run_with_context(
        self,
        user_request: str,
        workflow_path: str = None,
        lora_name: str = None,
    ) -> str:
        """
        Run the agent with additional context about the workflow and LoRA.

        This is called by the orchestrator to provide structured context
        along with the user's natural language request.
        """
        context_parts = [user_request]

        if workflow_path:
            context_parts.append(f"\n[컨텍스트] 사용할 워크플로우 파일: {workflow_path}")

        if lora_name:
            context_parts.append(f"[컨텍스트] 적용할 LoRA 모델: {lora_name}")

        context_parts.append(
            "\n위 정보를 바탕으로 워크플로우를 로드하고, 프롬프트를 검증한 후, "
            "이미지 생성을 실행해주세요."
        )

        full_request = "\n".join(context_parts)
        return self.run(full_request)
