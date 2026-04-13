# Qwen 9B Agent + ComfyUI Image Generation Pipeline

Qwen 9B를 에이전트로 설정하고, ComfyUI API를 도구(Tool)로 제공하여
프롬프트 검증부터 이미지 생성까지 하나의 파이프라인에서 처리하는 시스템입니다.

---

## 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────┐
│                      Orchestrator (main.py)                 │
│                                                             │
│  사용자 요청 수신 → Qwen 9B 에이전트에 전달 → 결과 반환       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Qwen 9B Agent (agent.py)                  │
│                                                             │
│  ReAct 루프: Think → Tool Call → Observe → Think → ...      │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ 프롬프트 검증 체크리스트                               │    │
│  │  1. 영어 작성 여부 (한국어 → 영어 번역)                │    │
│  │  2. 피사체/장면 명확성                                │    │
│  │  3. 스타일·조명·분위기 세부사항                        │    │
│  │  4. Negative prompt 적절성                           │    │
│  │  5. 프롬프트 길이 적정성                              │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  도구 호출 ──┬── load_workflow                               │
│              ├── set_lora                                    │
│              ├── set_prompt                                  │
│              ├── set_sampler_params                          │
│              ├── execute_workflow                            │
│              └── list_available_loras                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 ComfyUI API Server (:8188)                  │
│                                                             │
│  POST /prompt    ← 워크플로우 실행 요청                      │
│  GET  /history   ← 실행 상태 폴링                           │
│  GET  /view      ← 생성된 이미지 다운로드                    │
│  GET  /object_info ← 사용 가능 모델 조회                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 프로젝트 구조

```
comfy-qwen/
├── config.py                          # ComfyUI / Qwen 서버 엔드포인트 설정
├── comfyui_client.py                  # ComfyUI HTTP API 클라이언트
├── workflow_converter.py              # 프론트엔드 워크플로우 JSON → API 포맷 변환
├── tools.py                           # Qwen 에이전트 도구 정의 (스키마 + 구현)
├── agent.py                           # Qwen 9B 에이전트 (ReAct 루프, 시스템 프롬프트)
├── main.py                            # 오케스트레이터 진입점
├── requirements.txt                   # Python 의존성
├── flux_depth_lora_example.json       # ComfyUI 워크플로우 (Flux Depth LoRA)
└── README.md
```

---

## 에이전트 도구

| 도구 | 설명 |
|---|---|
| `load_workflow` | ComfyUI 워크플로우 JSON 파일을 로드하고 구조를 분석합니다. 다른 도구를 사용하기 전에 반드시 먼저 호출해야 합니다. |
| `set_lora` | 워크플로우에 적용할 LoRA 모델과 강도(strength)를 설정합니다. |
| `set_prompt` | 이미지 생성용 positive/negative 프롬프트를 설정합니다. |
| `set_sampler_params` | KSampler 파라미터를 조정합니다 (seed, steps, cfg, sampler, scheduler, denoise). |
| `execute_workflow` | 워크플로우를 ComfyUI 서버에 제출하고 이미지 생성 완료까지 대기합니다. |
| `list_available_loras` | ComfyUI 서버에서 사용 가능한 LoRA 모델 목록을 조회합니다. |

---

## 실행 흐름

```
오케스트레이터가 Qwen 9B에게 전달
  ├── 검증할 프롬프트
  ├── 사용할 ComfyUI 워크플로우 경로
  └── 적용할 LoRA 모델명

        ▼

Qwen 9B 에이전트 실행
  ├── Step 1. load_workflow     워크플로우 로드 및 분석
  ├── Step 2. (내부 추론)        프롬프트 검증, 한→영 번역, 품질 체크
  ├── Step 3. set_lora          LoRA 모델 확인/변경
  ├── Step 4. set_prompt        검증된 프롬프트 반영
  ├── Step 5. set_sampler_params 필요 시 파라미터 조정
  └── Step 6. execute_workflow  ComfyUI에 제출 → 이미지 생성 대기

        ▼

ComfyUI가 이미지 생성
  └── 결과를 오케스트레이터에 반환
```

---

## 워크플로우 정보 (flux_depth_lora_example.json)

| 항목 | 값 |
|---|---|
| Base Model | `flux1-dev-fp8.safetensors` (UNETLoader) |
| LoRA | `flux1-depth-dev-lora.safetensors` (strength: 1.0) |
| VAE | `ae.safetensors` |
| Text Encoders | `t5xxl_fp16.safetensors`, `clip_l.safetensors` |
| Sampler | euler / normal / steps=20 / cfg=1 / denoise=1.0 |
| Depth Model | `lotus-depth-d-v1-1.safetensors` (Lotus Depth) |
| Input Image | `flux_depth_lora_example_input_image.png` |

### 필요 모델 저장 위치

```
ComfyUI/
├── models/
│   ├── text_encoders/
│   │   ├── t5xxl_fp16.safetensors
│   │   └── clip_l.safetensors
│   ├── loras/
│   │   └── flux1-depth-dev-lora.safetensors
│   ├── diffusion_models/
│   │   ├── flux1-dev-fp8.safetensors
│   │   └── lotus-depth-d-v1-1.safetensors
│   └── vae/
│       ├── ae.safetensors
│       └── vae-ft-mse-840000-ema-pruned.safetensors
```

---

## 사전 요건

| 항목 | 내용 |
|---|---|
| **ComfyUI API 서버** | ComfyUI가 API 서버 모드로 실행되어 있어야 함 (기본 포트 8188) |
| **Ollama** | `ollama serve`로 실행 중이어야 함 (기본 포트 11434) |
| **Qwen 모델** | `ollama pull qwen3:8b`로 설치 |
| **Python** | 3.11 이상 권장 |
| **네트워크** | Qwen 서버(Ollama)와 ComfyUI 서버가 같은 네트워크에서 접근 가능해야 함 |

---

## 모델 다운로드

**필수:** Flux 및 관련 모델을 미리 다운로드해야 합니다.

### HuggingFace 인증 (필수 단계)

모든 모델이 HuggingFace의 gated 저장소에 있으므로, 먼저 인증이 필요합니다:

1. **HuggingFace 계정 생성**
   - https://huggingface.co/join 에서 계정 생성

2. **모델 액세스 승인**
   - 다음 저장소들을 방문하여 "Access repository" 클릭:
     - https://huggingface.co/black-forest-labs/FLUX.1-dev
     - https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev
     - https://huggingface.co/IDEA-Research/lotus-depth-v1

3. **액세스 토큰 생성**
   - https://huggingface.co/settings/tokens 방문
   - "New token" 클릭 (Role: read)
   - 토큰 복사

4. **HuggingFace CLI로 로그인**
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   ```
   - 토큰 붙여넣기
   - Enter 누르기

### 자동 다운로드 (권장)

```bash
# 위의 인증 단계 완료 후
python download_models.py --comfyui-path "C:/Users/joyco/OneDrive/Desktop/ComfyUI"
```

**주의:**
- 약 20GB의 디스크 공간 필요
- 네트워크 속도에 따라 수십 분 소요 가능
- 인증이 완료되어야 다운로드 가능

### 수동 다운로드 (선택사항)

각 모델을 HuggingFace에서 직접 다운로드한 후 위의 "필요 모델 저장 위치"에 맞게 배치하세요:

- **Base Model**: [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/flux1-dev-fp8.safetensors)
- **Depth LoRA**: [black-forest-labs/FLUX.1-Depth-dev](https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev/blob/main/flux1-depth-dev-lora.safetensors)
- **VAE**: [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/ae.safetensors)
- **Text Encoders**: [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) - `t5xxl_fp16.safetensors`, `clip_l.safetensors`
- **Depth Model**: [IDEA-Research/lotus-depth-v1](https://huggingface.co/IDEA-Research/lotus-depth-v1/blob/main/lotus-depth-d-v1-1.safetensors)

---

## 설치 및 실행

### comfy-qwen 에이전트 실행

```bash
# 1. comfy-qwen 디렉토리로 이동
cd C:/Users/joyco/OneDrive/Desktop/dana/comfy-qwen

# 2. 가상환경 활성화
.\.venv\Scripts\activate

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 모델 다운로드 (위의 "모델 다운로드" 섹션 참조)
python download_models.py --comfyui-path "C:/Users/joyco/OneDrive/Desktop/ComfyUI"
```

### ComfyUI 서버 시작 (별도 터미널)

```bash
# 1. ComfyUI 디렉토리로 이동
cd C:/Users/joyco/OneDrive/Desktop/ComfyUI

# 2. ComfyUI 가상환경 활성화
.\venv\Scripts\activate

# 3. ComfyUI 의존성 설치 (처음 1회만)
pip install -r requirements.txt

# 4. ComfyUI 서버 실행
python main.py --listen
```

### Ollama 서버 시작 (별도 터미널)

```bash
# Ollama 모델 풀 (처음 1회만)
ollama pull qwen3:8b

# Ollama 서버 시작
ollama serve
```

### 에이전트 실행 (comfy-qwen 터미널에서)

```bash
# 대화형 모드
python main.py

# 단일 프롬프트
python main.py --prompt "A cyberpunk city at night with neon lights"

# LoRA 명시 지정
python main.py --prompt "depth-aware portrait" --lora "flux1-depth-dev-lora.safetensors"

# 사전 체크 건너뛰기
python main.py --skip-check --prompt "A serene Japanese garden"
```

### 전체 시작 순서 요약

1. **터미널 1**: `ollama serve` (Ollama 서버 - 이미 실행 중)
2. **터미널 2**: ComfyUI 디렉토리에서 `.\venv\Scripts\activate` 후 `python main.py --listen` (ComfyUI 서버)
3. **터미널 3**: comfy-qwen 디렉토리에서 `.\.venv\Scripts\activate` 후 `python main.py` (에이전트)

---

## 설정 변경

[config.py](config.py)에서 서버 주소와 모델명을 변경할 수 있습니다:

```python
# ComfyUI 서버
COMFYUI_HOST = "127.0.0.1"
COMFYUI_PORT = 8188

# Qwen via Ollama
QWEN_BASE_URL = "http://127.0.0.1:11434/v1"
QWEN_MODEL = "qwen3:8b"
```

---

## 장점

- **파이프라인 단순화**: 기존에 오케스트레이터가 별도로 ComfyUI를 호출하던 것을 Qwen 9B 에이전트가 검증과 실행을 한 번에 처리
- **동적 LoRA 선택**: 씬의 분위기나 인물 특성에 따라 에이전트가 동적으로 LoRA 모델을 선택 가능
- **프롬프트 최적화**: 에이전트가 프롬프트를 자동으로 검증·번역·보강하여 생성 품질 향상
- **확장성**: 새로운 도구(업스케일, 인페인팅 등)를 tools.py에 추가하면 에이전트가 바로 활용 가능

## 고려사항

- 에이전트 도구 호출이 추가되므로 직접 API 호출 대비 약간의 레이턴시 증가
- Qwen 9B의 도구 호출 정확도는 모델 크기 특성상 가끔 재시도가 필요할 수 있음
- ComfyUI 워크플로우가 복잡해질수록 workflow_converter.py의 노드 타입 매핑 확장 필요
