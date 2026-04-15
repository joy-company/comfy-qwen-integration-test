# comfy-qwen

Gemini + Qwen + ComfyUI 기반 AI 이미지 생성 파이프라인

## 개요

**단일 프롬프트 모드** — 프롬프트를 입력하면 Qwen이 최적화하여 ComfyUI로 이미지 생성

**에피소드 모드** — 드라마 대본을 넣으면 모든 컷 이미지를 자동 생성:

```
사용자 입력 (에피소드 대본)
        │
        ▼
┌──────────────┐
│   GEMINI     │  1. 에피소드 바이블 생성 (캐릭터, 의상, 조명)
│              │  2. 대본을 프로덕션 컷으로 분할
│              │  3. 컷별 이미지 프롬프트 생성
└──────┬───────┘
       │ 프롬프트 + 바이블
       ▼
┌──────────────┐
│    QWEN      │  4. 바이블 기준 프롬프트 일관성 검증
│  (리뷰어)    │     - 의상, 시선, 손-물체, 아티팩트, 감정
└──────┬───────┘
       │ 승인? ──아니오──▶ Gemini에 피드백 전달 후 재생성
       │ 예
       ▼
┌──────────────┐
│   ComfyUI    │  5. 이미지 생성 큐 등록
│              │  6. output/ 폴더에 자동 저장
└──────────────┘
```

## 사전 요건

| 항목 | 설명 |
|------|------|
| ComfyUI | StabilityMatrix 또는 단독 실행 (포트 8188) |
| Ollama | `ollama serve` 실행 중, `qwen3:8b` 설치 완료 |
| Gemini API 키 | `.env` 파일에 설정 |

## 설치

```bash
cd C:\Users\joyco\OneDrive\Desktop\dana\comfy-qwen
.venv\Scripts\activate
pip install -r requirements.txt
```

`.env` 파일 생성:
```
GEMINI_API_KEY=your-key-here
```

## 실행 방법

**실행 전:** ComfyUI(StabilityMatrix)와 Ollama가 실행 중인지 확인

```bash
# 1. 터미널에서 프로젝트 폴더로 이동
cd C:\Users\joyco\OneDrive\Desktop\dana\comfy-qwen

# 2. 가상환경 활성화
.venv\Scripts\activate

# 3. 아래 모드 중 하나 선택:

# 대화형 모드 — 프롬프트를 하나씩 입력
python main.py

# 단일 프롬프트 — 한 장의 이미지 생성
python main.py --prompt "28yo Korean woman, cinematic, photorealistic"

# 에피소드 모드 — 대본에서 모든 컷 이미지 일괄 생성
python main.py --episode episode1.txt
```

## 프로젝트 구조

```
comfy-qwen/
├── main.py                  # 진입점 (--prompt, --episode, 대화형)
├── config.py                # 서버 URL, 모델명, API 키 설정
├── .env                     # Gemini API 키 (git 미포함)
│
├── agent.py                 # 단일 프롬프트용 Qwen 에이전트
├── tools.py                 # ComfyUI 도구 (로드, 프롬프트 설정, 실행 등)
├── comfyui_client.py        # ComfyUI HTTP API 클라이언트
├── workflow_converter.py    # 프론트엔드 워크플로우 JSON → API 포맷 변환
│
├── gemini_client.py         # Gemini API: 바이블, 컷 분할, 프롬프트 생성
├── reviewer.py              # Qwen 기반 프롬프트 리뷰어 (일관성 검증)
├── episode_pipeline.py      # 에피소드 모드 비동기 오케스트레이터
│
├── flux_depth_lora_example.json  # ComfyUI 워크플로우 (FLUX + LoRA)
├── episode1.txt             # 1화 대본 예시
├── output/                  # 생성된 이미지 저장 폴더
└── requirements.txt
```

## 워크플로우 정보

| 항목 | 값 |
|------|---|
| 베이스 모델 | `flux1-dev.safetensors` (fp8) |
| LoRA | `xiaoshazi.safetensors` (강도 1.5) |
| VAE | `ae.safetensors` |
| CLIP | `clip_l.safetensors`, `t5xxl_fp16.safetensors` |
| 해상도 | 1024x1024 |

## 설정

`config.py`에서 변경 가능:

```python
COMFYUI_BASE_URL = "http://127.0.0.1:8188"
QWEN_BASE_URL = "http://127.0.0.1:11434/v1"
QWEN_MODEL = "qwen3:8b"
GEMINI_MODEL = "gemini-2.5-flash"
```
