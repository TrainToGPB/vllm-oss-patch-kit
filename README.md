# vllm-oss-patch-kit

GPT-OSS-120B 서빙을 위한 vLLM Python 패치 키트.

pip wheel의 최적화된 CUDA 커널을 유지하면서 Jinja2 chat template + Harmony 출력 파싱 패치만 적용합니다.

> **왜 소스 빌드를 안 하나요?**
> vLLM 소스 빌드는 모든 CUDA 아키텍처 대상으로 커널을 컴파일하여 FlashAttention 등의 런타임 성능이 pip wheel 대비 최대 3배 저하됩니다 (동시 요청 16개 기준). 이 키트는 pip wheel의 최적화된 바이너리를 그대로 사용합니다.

## Quick Start

```bash
# 새 서버에서 처음부터 설치
./apply.sh /path/to/venv --install

# 기존 vLLM 환경에 패치만 적용
source /path/to/venv/bin/activate
./apply.sh

# 특정 virtualenv 지정
./apply.sh /path/to/venv
```

## 적용되는 패치

| 패치 | 대상 파일 | 내용 |
|---|---|---|
| `serving.patch` | `chat_completion/serving.py` | `use_harmony` / `use_harmony_for_rendering` 분리, analysis channel prefix pre-feed |
| `harmony_utils.patch` | `parser/harmony_utils.py` | `sanitize_harmony_tool_name()`, `strip_harmony_control_tokens()` 추가 |
| `chat_utils.patch` | `chat_utils.py` | `thinking` 필드 → reasoning 매핑 |

## 디렉토리 구조

```
vllm-oss-patch-kit/
├── apply.sh                  # 패치 적용 스크립트
├── README.md
├── patches/
│   ├── serving.patch         # serving.py 패치
│   ├── harmony_utils.patch   # harmony_utils.py 패치
│   ├── chat_utils.patch      # chat_utils.py 패치
│   └── stream_profiler.py    # 스트리밍 성능 프로파일러 (선택)
└── templates/
    └── chat_template.jinja   # Jinja2 chat template
```

## 서버 실행

```bash
source /path/to/venv/bin/activate

vllm serve openai/gpt-oss-120b \
	--tensor-parallel-size 2 \
	--async-scheduling \
	--enable-prefix-caching \
	--max-cudagraph-capture 2048 \
	--max-num-batched-tokens 8192 \
	--stream-interval 20 \
	--chat-template '/data/users/shkim/vllm/scripts/chat_template.jinja' \
	--enable-auto-tool-choice \
	--tool-call-parser openai \
	--port 20009
```

## 프로파일러

스트리밍 성능 병목 분석이 필요할 때:

```bash
VLLM_STREAM_PROFILE=1 vllm serve openai/gpt-oss-120b ...
```

요청별로 `engine_wait_ms`, `py_process_ms`, `analysis_tokens`, `final_tokens` 등을 JSON으로 로깅합니다.

## 요구사항

- Python 3.12+
- CUDA 12.x
- vLLM 0.15.0 (자동 설치/재설치 지원)
