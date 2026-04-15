"""
Async episode pipeline.

Flow:
1. Gemini generates Episode Bible from script
2. Gemini divides script into cuts
3. For each cut (processed concurrently):
   a. Gemini generates image prompt
   b. Qwen reviews prompt against bible
   c. If rejected → Gemini revises (up to MAX_REVIEW_RETRIES)
   d. If approved → queue to ComfyUI
4. All images auto-saved to output/
"""

import json
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gemini_client
from reviewer import PromptReviewer
from comfyui_client import ComfyUIClient
from workflow_converter import load_frontend_workflow, convert_to_api_format
from config import DEFAULT_WORKFLOW_PATH, MAX_REVIEW_RETRIES


OUTPUT_DIR = Path(__file__).parent / "output"


class EpisodePipeline:
    def __init__(self, max_concurrent: int = 3):
        self.reviewer = PromptReviewer()
        self.comfyui = ComfyUIClient()
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self.bible: dict = {}
        self.cuts: list[dict] = []
        self.results: list[dict] = []

    @staticmethod
    def _fmt_elapsed(seconds: float) -> str:
        """Format elapsed seconds as '12.3s' or '2m 15s'."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"

    async def run(self, script: str) -> list[dict]:
        """Run the full episode pipeline."""
        loop = asyncio.get_event_loop()
        total_start = time.time()

        # --- Phase 1: Generate Bible ---
        print("\n" + "=" * 60)
        print("[Phase 1] Gemini: 에피소드 바이블 생성 중...")
        print("=" * 60)

        t0 = time.time()
        self.bible = await loop.run_in_executor(
            self.executor, gemini_client.generate_bible, script
        )
        t1 = time.time()
        print(f"  ✓ 바이블 생성 완료: {len(self.bible.get('characters', {}))} 캐릭터, "
              f"{len(self.bible.get('scenes', {}))} 씬")

        bible_path = OUTPUT_DIR / "episode_bible.json"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        bible_path.write_text(
            json.dumps(self.bible, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  ✓ 바이블 저장: {bible_path}")
        print(f"  ⏱ Phase 1 완료: {self._fmt_elapsed(t1 - t0)}")

        # --- Phase 2: Divide into Cuts ---
        print("\n" + "=" * 60)
        print("[Phase 2] Gemini: 대본을 컷으로 분할 중...")
        print("=" * 60)

        t0 = time.time()
        self.cuts = await loop.run_in_executor(
            self.executor, gemini_client.divide_into_cuts, script, self.bible
        )
        t1 = time.time()
        print(f"  ✓ 총 {len(self.cuts)}개 컷 생성")

        cuts_path = OUTPUT_DIR / "episode_cuts.json"
        cuts_path.write_text(
            json.dumps(self.cuts, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  ✓ 컷 목록 저장: {cuts_path}")
        print(f"  ⏱ Phase 2 완료: {self._fmt_elapsed(t1 - t0)}")

        # --- Phase 3: Generate, Review, Queue (async per cut) ---
        print("\n" + "=" * 60)
        print(f"[Phase 3] 컷별 프롬프트 생성 → 리뷰 → 이미지 생성 ({len(self.cuts)}개)")
        print("=" * 60)

        t0 = time.time()
        tasks = [
            self._process_cut(loop, i, cut)
            for i, cut in enumerate(self.cuts)
        ]
        self.results = await asyncio.gather(*tasks)
        t1 = time.time()

        # --- Summary ---
        succeeded = sum(1 for r in self.results if r.get("status") == "success")
        failed = sum(1 for r in self.results if r.get("status") != "success")
        total_elapsed = time.time() - total_start

        print("\n" + "=" * 60)
        print(f"[완료] 총 {len(self.results)}컷: {succeeded} 성공, {failed} 실패")
        print(f"  ⏱ Phase 3: {self._fmt_elapsed(t1 - t0)}")
        print(f"  ⏱ 전체 소요: {self._fmt_elapsed(total_elapsed)}")
        print("=" * 60)

        # Save full results
        results_path = OUTPUT_DIR / "episode_results.json"
        results_path.write_text(
            json.dumps(self.results, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        return self.results

    async def _process_cut(self, loop, index: int, cut: dict) -> dict:
        """Process a single cut: generate prompt → review → queue."""
        cut_num = cut.get("cut_number", index + 1)
        scene_id = cut.get("scene_id", "?")
        prefix = f"  [컷 {cut_num}/{len(self.cuts)}] ({scene_id})"

        # Step 1: Generate image prompt
        print(f"{prefix} 프롬프트 생성 중...")
        image_prompt = await loop.run_in_executor(
            self.executor,
            gemini_client.generate_image_prompt,
            cut, self.bible,
        )

        # Step 2: Review loop
        approved = False
        for attempt in range(1, MAX_REVIEW_RETRIES + 1):
            print(f"{prefix} Qwen 리뷰 중... (시도 {attempt}/{MAX_REVIEW_RETRIES})")
            review = await loop.run_in_executor(
                self.executor,
                self.reviewer.review,
                image_prompt, cut, self.bible,
            )

            if review.get("status") == "approved":
                print(f"{prefix} ✓ 승인됨: {review.get('reason', '')[:80]}")
                approved = True
                break

            # Rejected — revise
            feedback = review.get("reason", "") + " " + " ".join(review.get("issues", []))
            print(f"{prefix} ✗ 거부: {feedback[:100]}")

            image_prompt = await loop.run_in_executor(
                self.executor,
                gemini_client.revise_image_prompt,
                image_prompt, cut, self.bible, feedback,
            )
            print(f"{prefix} 프롬프트 수정 완료")

        if not approved:
            print(f"{prefix} ⚠ 최대 리뷰 횟수 도달 — 현재 프롬프트로 진행")

        # Step 3: Queue image generation
        print(f"{prefix} ComfyUI 큐 등록 중...")
        result = await loop.run_in_executor(
            self.executor,
            self._queue_image, image_prompt, cut_num,
        )

        if result.get("status") == "success":
            saved = result.get("saved_to", [])
            print(f"{prefix} ✓ 이미지 생성 완료: {saved}")
        else:
            print(f"{prefix} ✗ 이미지 생성 실패: {result.get('error', '?')}")

        return {
            "cut_number": cut_num,
            "scene_id": scene_id,
            "prompt": image_prompt,
            "review_status": "approved" if approved else "forced",
            **result,
        }

    def _queue_image(self, prompt: str, cut_number: int) -> dict:
        """Load workflow, set prompt, execute, save image."""
        # Load a fresh workflow for each cut
        wf_path = Path(DEFAULT_WORKFLOW_PATH)
        if not wf_path.exists():
            return {"status": "error", "error": f"워크플로우 없음: {wf_path}"}

        from workflow_converter import load_frontend_workflow, convert_to_api_format, extract_workflow_info

        frontend_wf = load_frontend_workflow(wf_path)
        api_wf = convert_to_api_format(frontend_wf)
        wf_info = extract_workflow_info(frontend_wf)

        # Set positive prompt
        pos_info = wf_info.get("positive_prompt")
        if pos_info:
            node_id = pos_info["node_id"]
            if node_id in api_wf:
                api_wf[node_id]["inputs"]["text"] = prompt

        # Set negative prompt
        neg_info = wf_info.get("negative_prompt")
        if neg_info:
            node_id = neg_info["node_id"]
            if node_id in api_wf:
                api_wf[node_id]["inputs"]["text"] = (
                    "low quality, blurry, deformed, bad anatomy, bad hands, "
                    "extra fingers, watermark, text, cartoon, anime"
                )

        # Submit
        try:
            prompt_id = self.comfyui.queue_prompt(api_wf)
        except Exception as e:
            return {"status": "error", "error": str(e)}

        # Wait for completion
        try:
            history = self.comfyui.wait_for_completion(prompt_id)
        except TimeoutError as e:
            return {"status": "error", "error": str(e)}

        # Save images
        images = self.comfyui.get_output_images(history)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        saved_paths = []
        for img_info in images:
            img_data = self.comfyui.get_image(
                img_info["filename"],
                img_info.get("subfolder", ""),
                img_info.get("type", "output"),
            )
            # Name by cut number for easy identification
            out_name = f"cut_{cut_number:03d}_{img_info['filename']}"
            out_path = OUTPUT_DIR / out_name
            out_path.write_bytes(img_data)
            saved_paths.append(str(out_path))

        return {
            "status": "success",
            "prompt_id": prompt_id,
            "saved_to": saved_paths,
        }


def run_episode(script: str) -> list[dict]:
    """Entry point — run the full episode pipeline synchronously."""
    pipeline = EpisodePipeline(max_concurrent=2)
    return asyncio.run(pipeline.run(script))
