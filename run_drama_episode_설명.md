# Run Drama Episode 설명서

## 개요

`run_drama_episode.py`는 대본 파일 하나로 **드라마 분석부터 이미지 생성까지** 전체 파이프라인을 자동 실행하는 통합 진입점입니다.

## 실행 방법

```powershell
# 1. 가상환경 활성화
.\.venv\Scripts\Activate

# 2. 실행
python -u run_drama_episode.py <대본파일> <에피소드번호>

# 예시
python -u run_drama_episode.py episode1.txt 1
```

### 사전 요구사항

| 항목 | 확인 방법 |
|------|----------|
| Ollama 실행 중 | `ollama serve` |
| Gemma 4 모델 설치 | `ollama pull gemma4:26b` |
| Qwen 모델 설치 | `ollama pull qwen3:8b` |
| ComfyUI 실행 중 | `http://localhost:8188` 접속 확인 |
| Gemini API 키 | `.env` 파일에 `GEMINI_API_KEY=...` 설정 |
| 가상환경 활성화 | `.\.venv\Scripts\Activate` |

## 전체 파이프라인 흐름

```
대본 파일 (episode1.txt)
    │
    ▼
╔══════════════════════════════════════════════════════╗
║  Stage 1: 드라마 분석 (drama_pipeline.js)            ║
║  Node.js 실행                                        ║
║                                                      ║
║  ① Script Analyst    (Gemma 4, 로컬)  → YAML 분석    ║
║  ② Director Agent    (Gemma 4, 로컬)  → YAML 비트    ║
║  ③ Cinematography    (Gemma 4, 로컬)  → YAML 샷리스트 ║
║  ④ Final Arbiter     (Gemini 클라우드) → JSON 컷플랜  ║
║     └─ Gemini 실패 시 → Gemma 4 폴백                  ║
║                                                      ║
║  출력: output/ep{N}/drama_cuts.json                   ║
╚══════════════════════════╤═══════════════════════════╝
                           │
                           ▼
╔══════════════════════════════════════════════════════╗
║  Stage 2: 이미지 생성 (episode_pipeline.py)           ║
║  Python 실행                                         ║
║                                                      ║
║  Phase 1: Gemini로 에피소드 바이블 생성                ║
║           → 캐릭터 DNA, 의상, 조명, 지속 아티팩트       ║
║           → bible.json 저장                           ║
║                                                      ║
║  Phase 2: drama_cuts.json 로드 (Gemini 컷 분할 생략)  ║
║                                                      ║
║  Phase 3: 컷별 처리 (동시 2개)                        ║
║     각 컷마다:                                        ║
║     a. Gemini: 드라마 컷 설명 + 바이블 → 이미지 프롬프트 ║
║     b. Qwen: 일관성 리뷰 (의상, 시선, 감정, 프레이밍)   ║
║        └─ 거부 시 → Gemini가 수정 (최대 3회)           ║
║     c. LoRA 선택 (캐릭터별 매핑)                       ║
║     d. ComfyUI: 이미지 생성 → PNG 저장                 ║
║                                                      ║
║  출력: output/ep{N}/ 폴더                             ║
╚══════════════════════════════════════════════════════╝
```

## Stage 1 상세: 드라마 분석

`run_drama_episode.py`가 `drama_pipeline.js`를 Node.js 자식 프로세스로 실행합니다.

- 대본 파일을 UTF-8로 읽어 stdin으로 전달
- 자동으로 `---END---` 종료 표시 추가
- `--episode` 인자로 에피소드 번호 전달

### 3개 에이전트 (순차 실행, Gemma 4 로컬)

| 에이전트 | 온도 | 역할 | 출력 |
|---------|------|------|------|
| Script Analyst | 0.3 | 소품/캐스트/의상 분류, 감정 엔진 식별 | YAML |
| Director Agent | 0.8 | 드라마틱 비트 분할, 전술/강도 매핑 | YAML |
| Cinematography | 0.5 | 렌즈/조리개/조명/카메라 움직임 설계 | YAML |

### Final Arbiter (Gemini 클라우드, 폴백: Gemma 4)

3개 에이전트의 분석을 종합하여 **구조화된 JSON**으로 출력:

```json
{
  "validation_issues": [...],
  "cuts": [
    {
      "cut_number": 1,
      "scene_id": "S#1",
      "reference": "대본 원문 인용",
      "description": "컷에 보이는 내용",
      "technical_specs": { "lens", "aperture", "iso", "depth_of_field" },
      "lighting": { "direction", "style", "color" },
      "composition": { "shot_type", "focal_point", "camera_movement", ... },
      "emotional_intent": "감정적 목적",
      "pacing_note": "리듬 노트",
      "estimated_duration": "2.5s"
    }
  ],
  "production_notes": { ... }
}
```

## Stage 2 상세: 이미지 생성

### Phase 1: 바이블 생성

Gemini가 대본을 분석하여 에피소드 바이블을 생성합니다:
- **캐릭터 DNA**: 얼굴 구조, 체형, 습관적 표정
- **씬별 의상**: 정확한 소재, 색상, 디자인
- **지속 아티팩트**: 와인 얼룩, 따귀 자국 등 누적 변화
- **조명 환경**: 키라이트 방향, 색온도, 그림자 강도

### Phase 2: 드라마 컷 로드

Stage 1에서 생성된 `drama_cuts.json`을 읽어옵니다. 기존 파이프라인의 Gemini 컷 분할 단계를 건너뜁니다.

### Phase 3: 컷별 이미지 생성

각 드라마 컷에 대해:

1. **이미지 프롬프트 생성** (Gemini)
   - 드라마 컷의 description, technical_specs, lighting, composition, emotional_intent를 바이블의 캐릭터 DNA/의상과 결합
   - "Korean man/woman" 명시, 정확한 의상, 카메라 프레이밍, 품질 태그 포함

2. **일관성 리뷰** (Qwen 9B, 로컬)
   - 6가지 검증: 의상, 지속 아티팩트, 시선/방향, 손-물체, 감정 강도, 프레이밍
   - 거부 시 Gemini가 수정 (최대 3회)

3. **LoRA 선택**
   - `config.py`의 `CHARACTER_LORA` 매핑에서 캐릭터별 LoRA 파일 선택
   - 매핑에 없으면 `DEFAULT_LORA` 사용

4. **ComfyUI 이미지 생성**
   - 워크플로우 로드 → 프롬프트/LoRA 설정 → 큐 등록 → 완료 대기 → PNG 저장

## 출력 파일 구조

```
output/ep1/
├── drama_cuts.json    # Stage 1 결과: 구조화된 컷 플랜
├── script.txt         # 원본 대본
├── bible.json         # Phase 1: 에피소드 바이블
├── cuts.json          # Phase 2: 드라마 컷 복사본
├── prompts.json       # Phase 3: 컷별 최종 이미지 프롬프트
├── results.json       # Phase 3: 전체 실행 결과
├── cut_001.png        # 생성된 이미지
├── cut_002.png
├── cut_003.png
└── ...
```

## 모델 사용 정리

| 단계 | 모델 | 위치 | 용도 |
|------|------|------|------|
| 에이전트 분석 | Gemma 4 26B | Ollama 로컬 | 대본 → 드라마 컷 분석 |
| Final Arbiter | Gemini 2.5 Flash | Google 클라우드 | 에이전트 종합 → JSON 컷플랜 |
| Final Arbiter 폴백 | Gemma 4 26B | Ollama 로컬 | Gemini 503 시 자동 전환 |
| 바이블 생성 | Gemini | Google 클라우드 | 대본 → 캐릭터/씬 데이터 |
| 이미지 프롬프트 | Gemini | Google 클라우드 | 드라마 컷 + 바이블 → 프롬프트 |
| 프롬프트 수정 | Gemini | Google 클라우드 | 리뷰 피드백 반영 |
| 일관성 리뷰 | Qwen 3 8B | Ollama 로컬 | 프롬프트 ↔ 바이블 검증 |
| 이미지 생성 | Flux (LoRA) | ComfyUI 로컬 | 프롬프트 → PNG |

## 에러 처리

| 에러 | 원인 | 대응 |
|------|------|------|
| Gemini 503 (Final Arbiter) | Google 서버 과부하 | 3회 재시도 후 Gemma 4 폴백 |
| Ollama fetch failed | 모델 콜드스타트 중 연결 끊김 | 10초 대기 후 1회 재시도 |
| Ollama timeout | 응답 시간 초과 (10분) | 타임아웃 값 증가 또는 작은 모델 사용 |
| drama_cuts.json 미생성 | Final Arbiter가 유효한 JSON 미생성 | 재실행 필요 |
| Qwen 리뷰 거부 3회 | 프롬프트가 바이블과 불일치 | 현재 프롬프트로 강제 진행 |
