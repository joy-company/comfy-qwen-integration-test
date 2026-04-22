"""
Drama Episode Pipeline — full flow entry point.

Orchestrates the complete pipeline:
1. Read script from file
2. Run drama_pipeline.js (3 Gemma agents + Gemini arbiter) → drama_cuts.json
3. Run episode_pipeline.py with drama cuts (bible + image prompts + Qwen review + ComfyUI)

Usage:
    python run_drama_episode.py <script_file> <episode_number>

Example:
    python run_drama_episode.py scripts/ep3.txt 3
"""

import sys
import json
import asyncio
import subprocess
from pathlib import Path

from episode_pipeline import EpisodePipeline


def run_drama_analysis(script_path: str, episode_num: str) -> Path:
    """Run drama_pipeline.js and return the path to drama_cuts.json."""
    drama_script = Path(__file__).parent / "drama_pipeline.js"
    output_path = Path(__file__).parent / "output" / f"ep{episode_num}" / "drama_cuts.json"

    print("\n" + "=" * 60)
    print("[Stage 1] 드라마 파이프라인 실행 중 (Gemma 4 agents + Gemini arbiter)...")
    print("=" * 60)

    # Pipe the script file content to drama_pipeline.js via stdin
    script_content = Path(script_path).read_text(encoding="utf-8")
    # Append the sentinel line so drama_pipeline.js knows input is complete
    stdin_input = script_content + "\n---END---\n"

    result = subprocess.run(
        ["node", str(drama_script), "--episode", episode_num],
        input=stdin_input,
        capture_output=False,
        text=True,
        encoding="utf-8",
        cwd=str(Path(__file__).parent),
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
    )

    if result.returncode != 0:
        print(f"  ✗ drama_pipeline.js failed (exit code {result.returncode})")
        sys.exit(1)

    if not output_path.exists():
        print(f"  ✗ drama_cuts.json not found at {output_path}")
        print("    The Final Arbiter may have failed to produce valid JSON.")
        sys.exit(1)

    # Verify it's valid JSON with cuts
    data = json.loads(output_path.read_text(encoding="utf-8"))
    num_cuts = len(data.get("cuts", []))
    print(f"\n  ✓ 드라마 분석 완료: {num_cuts}개 컷 생성됨")
    print(f"  ✓ 저장: {output_path}")

    return output_path


def run_image_pipeline(script_path: str, drama_cuts_path: Path, episode_num: str):
    """Run the image generation pipeline with drama cuts."""
    print("\n" + "=" * 60)
    print("[Stage 2] 이미지 생성 파이프라인 실행 중...")
    print("=" * 60)

    script_content = Path(script_path).read_text(encoding="utf-8")
    pipeline = EpisodePipeline(max_concurrent=2)
    results = asyncio.run(
        pipeline.run_with_drama_cuts(script_content, str(drama_cuts_path), episode_num)
    )

    return results


def main():
    if len(sys.argv) < 3:
        print("Usage: python run_drama_episode.py <script_file> <episode_number>")
        print("Example: python run_drama_episode.py scripts/ep3.txt 3")
        sys.exit(1)

    script_path = sys.argv[1]
    episode_num = sys.argv[2]

    if not Path(script_path).exists():
        print(f"Error: Script file not found: {script_path}")
        sys.exit(1)

    print("=" * 60)
    print("  DRAMA EPISODE PIPELINE")
    print(f"  Script: {script_path}")
    print(f"  Episode: {episode_num}")
    print("=" * 60)

    # Stage 1: Drama analysis (Node.js → Gemma 4 + Gemini)
    drama_cuts_path = run_drama_analysis(script_path, episode_num)

    # Stage 2: Image generation (Python → Gemini + Qwen + ComfyUI)
    results = run_image_pipeline(script_path, drama_cuts_path, episode_num)

    # Final summary
    succeeded = sum(1 for r in results if r.get("status") == "success")
    failed = sum(1 for r in results if r.get("status") != "success")

    print("\n" + "=" * 60)
    print(f"  COMPLETE: {succeeded} images generated, {failed} failed")
    print(f"  Output: output/ep{episode_num}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
