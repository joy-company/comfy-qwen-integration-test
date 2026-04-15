"""
Orchestrator: Qwen 9B Agent + ComfyUI Pipeline

Usage:
    # Interactive mode
    python main.py

    # Single request
    python main.py --prompt "A cyberpunk city at night with neon lights"

    # Episode mode — generate all cut images from a script file
    python main.py --episode script.txt

    # With explicit LoRA and workflow
    python main.py --prompt "A portrait in oil painting style" \
                   --workflow flux_depth_lora_example.json \
                   --lora "flux1-depth-dev-lora.safetensors"
"""

import argparse
import sys
from pathlib import Path

from agent import QwenAgent
from comfyui_client import ComfyUIClient
from config import COMFYUI_BASE_URL, DEFAULT_WORKFLOW_PATH


def check_prerequisites():
    """Verify that required services are accessible."""
    print("[체크] 사전 요건 확인 중...")

    # Check ComfyUI
    client = ComfyUIClient()
    if client.is_alive():
        print("  ✓ ComfyUI 서버 연결 성공")
    else:
        print(f"  ✗ ComfyUI 서버에 연결할 수 없습니다 ({COMFYUI_BASE_URL})")
        print("    ComfyUI를 API 모드로 실행하세요: python main.py --listen")
        return False

    # Check workflow file exists
    wf_path = Path(DEFAULT_WORKFLOW_PATH)
    if wf_path.exists():
        print(f"  ✓ 워크플로우 파일 확인: {wf_path}")
    else:
        print(f"  ✗ 워크플로우 파일을 찾을 수 없습니다: {wf_path}")
        return False

    # Check Qwen (try a simple call)
    try:
        agent = QwenAgent()
        agent.client.models.list()
        print(f"  ✓ Qwen 모델 서버 연결 성공")
    except Exception as e:
        print(f"  ✗ Qwen 모델 서버에 연결할 수 없습니다: {e}")
        print("    Ollama가 실행 중인지 확인하세요: ollama serve")
        print("    Qwen 모델이 설치되어 있는지 확인하세요: ollama pull qwen3:8b")
        return False

    print()
    return True


def run_single(prompt: str, workflow: str = None, lora: str = None):
    """Run a single image generation request through the agent."""
    agent = QwenAgent()
    result = agent.run_with_context(
        user_request=prompt,
        workflow_path=workflow or DEFAULT_WORKFLOW_PATH,
        lora_name=lora,
    )
    print("\n" + "=" * 60)
    print("[완료] 에이전트 실행 결과:")
    print(result)
    return result


def run_interactive():
    """Run the agent in interactive mode."""
    print("=" * 60)
    print("  Qwen 9B + ComfyUI 이미지 생성 에이전트")
    print("  명령어: quit/exit - 종료, help - 도움말")
    print("=" * 60)

    agent = QwenAgent()

    while True:
        try:
            user_input = input("\n[프롬프트] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("종료합니다.")
            break
        if user_input.lower() == "help":
            print("""
사용법:
  이미지 생성 프롬프트를 입력하세요 (한국어 또는 영어).

  에이전트가 자동으로:
  1. 프롬프트를 검증하고 최적화합니다
  2. 워크플로우를 로드합니다
  3. 적절한 LoRA 모델을 적용합니다
  4. ComfyUI에서 이미지를 생성합니다

예시:
  > A futuristic robot standing in a garden
  > 네온 불빛이 빛나는 사이버펑크 도시의 야경
  > A serene Japanese garden with cherry blossoms
""")
            continue

        agent.run_with_context(
            user_request=user_input,
            workflow_path=DEFAULT_WORKFLOW_PATH,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Qwen 9B Agent + ComfyUI Image Generation Pipeline"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="이미지 생성 프롬프트 (지정하지 않으면 대화형 모드)",
    )
    parser.add_argument(
        "--episode", "-e",
        type=str,
        help="에피소드 대본 파일 경로 (컷 분할 → 이미지 일괄 생성)",
    )
    parser.add_argument(
        "--workflow", "-w",
        type=str,
        default=DEFAULT_WORKFLOW_PATH,
        help=f"ComfyUI 워크플로우 JSON 파일 경로 (기본값: {DEFAULT_WORKFLOW_PATH})",
    )
    parser.add_argument(
        "--lora", "-l",
        type=str,
        help="사용할 LoRA 모델명 (지정하지 않으면 워크플로우 기본값 사용)",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="사전 요건 확인 건너뛰기",
    )

    args = parser.parse_args()

    if not args.skip_check:
        if not check_prerequisites():
            print("\n사전 요건을 충족하지 못했습니다. --skip-check로 건너뛸 수 있습니다.")
            sys.exit(1)

    if args.episode:
        from episode_pipeline import run_episode
        script_path = Path(args.episode)
        if not script_path.exists():
            print(f"대본 파일을 찾을 수 없습니다: {script_path}")
            sys.exit(1)
        script_text = script_path.read_text(encoding="utf-8")
        print(f"[에피소드 모드] 대본 로드: {script_path} ({len(script_text)} 글자)")
        run_episode(script_text)
    elif args.prompt:
        run_single(args.prompt, args.workflow, args.lora)
    else:
        run_interactive()


if __name__ == "__main__":
    main()
