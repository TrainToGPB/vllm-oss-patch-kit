#!/bin/bash
set -e

# ============================================================
# vllm-oss-patch-kit: apply.sh
#
# GPT-OSS-120B를 위한 vLLM 패치 자동 적용 스크립트
# pip wheel의 최적화된 CUDA 커널을 유지하면서 Python 패치만 적용
#
# Usage:
#   ./apply.sh                          # 현재 활성화된 virtualenv에 적용
#   ./apply.sh /path/to/venv            # 지정된 virtualenv에 적용
#   ./apply.sh /path/to/venv --install  # venv 생성 + vllm 설치 + 패치
# ============================================================

EXPECTED_VLLM="0.15.0"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PATCH_DIR="$SCRIPT_DIR/patches"
TEMPLATE_DIR="$SCRIPT_DIR/templates"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err()  { echo -e "${RED}[✗]${NC} $1"; }

# ----------------------------------------------------------
# Parse args
# ----------------------------------------------------------
VENV_DIR=""
DO_INSTALL=false

for arg in "$@"; do
    case "$arg" in
        --install) DO_INSTALL=true ;;
        -*) echo "Unknown option: $arg"; exit 1 ;;
        *) VENV_DIR="$arg" ;;
    esac
done

# ----------------------------------------------------------
# 1. Resolve Python environment
# ----------------------------------------------------------
if [ -n "$VENV_DIR" ]; then
    if [ "$DO_INSTALL" = true ] && [ ! -d "$VENV_DIR" ]; then
        log "Creating virtualenv at $VENV_DIR ..."
        python3 -m venv "$VENV_DIR"
    fi

    if [ ! -f "$VENV_DIR/bin/activate" ]; then
        err "Not a valid virtualenv: $VENV_DIR"
        exit 1
    fi
    source "$VENV_DIR/bin/activate"
    log "Activated virtualenv: $VENV_DIR"
elif [ -z "$VIRTUAL_ENV" ]; then
    err "No virtualenv active and no path given."
    echo "Usage: $0 [/path/to/venv] [--install]"
    exit 1
else
    log "Using active virtualenv: $VIRTUAL_ENV"
fi

PYTHON="$(which python3 || which python)"
PIP="$PYTHON -m pip"

# ----------------------------------------------------------
# 2. Check / install vLLM version
# ----------------------------------------------------------
CURRENT_VLLM=$($PYTHON -c "
try:
    import vllm; print(vllm.__version__)
except ImportError:
    print('NOT_INSTALLED')
" 2>/dev/null)

if [ "$CURRENT_VLLM" = "NOT_INSTALLED" ]; then
    if [ "$DO_INSTALL" = true ]; then
        log "vLLM not found. Installing vllm==$EXPECTED_VLLM ..."
        $PIP install --upgrade pip -q
        $PIP install "vllm==$EXPECTED_VLLM" -q
        log "Installed vllm==$EXPECTED_VLLM"
    else
        err "vLLM is not installed. Run with --install to auto-install."
        exit 1
    fi
elif [ "$CURRENT_VLLM" != "$EXPECTED_VLLM" ]; then
    warn "vLLM version mismatch: $CURRENT_VLLM (expected $EXPECTED_VLLM)"

    if [ "$DO_INSTALL" = true ]; then
        log "Reinstalling vllm==$EXPECTED_VLLM ..."
        $PIP install "vllm==$EXPECTED_VLLM" --force-reinstall -q
        log "Reinstalled vllm==$EXPECTED_VLLM"
    else
        err "Run with --install to auto-reinstall, or fix manually."
        exit 1
    fi
else
    log "vLLM version OK: $CURRENT_VLLM"
fi

# ----------------------------------------------------------
# 3. Locate site-packages
# ----------------------------------------------------------
SITE=$($PYTHON -c "import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))")
log "site-packages: $SITE"

# Verify import path is site-packages (not editable install)
VLLM_PATH=$($PYTHON -c "import vllm; print(vllm.__file__)")
if [[ "$VLLM_PATH" != *"site-packages"* ]]; then
    err "vLLM is imported from $VLLM_PATH (editable install?)"
    err "This kit requires pip-installed vLLM. Uninstall editable install first."
    exit 1
fi

# ----------------------------------------------------------
# 4. Apply patches
# ----------------------------------------------------------
log "Applying patches..."
cd "$SITE"

APPLIED=0
SKIPPED=0
for p in serving.patch harmony_utils.patch chat_utils.patch; do
    if [ ! -f "$PATCH_DIR/$p" ]; then
        warn "Patch not found: $p"
        continue
    fi

    if patch -p1 --forward --dry-run < "$PATCH_DIR/$p" > /dev/null 2>&1; then
        patch -p1 --forward < "$PATCH_DIR/$p" > /dev/null 2>&1
        log "  Applied: $p"
        APPLIED=$((APPLIED + 1))
    else
        if patch -p1 --reverse --dry-run < "$PATCH_DIR/$p" > /dev/null 2>&1; then
            warn "  Already applied: $p"
            SKIPPED=$((SKIPPED + 1))
        else
            err "  Failed to apply: $p (manual conflict resolution needed)"
            exit 1
        fi
    fi
done

# ----------------------------------------------------------
# 5. Copy stream profiler (optional)
# ----------------------------------------------------------
PROFILER_DST="$SITE/vllm/entrypoints/openai/chat_completion/stream_profiler.py"
if [ -f "$PATCH_DIR/stream_profiler.py" ]; then
    cp "$PATCH_DIR/stream_profiler.py" "$PROFILER_DST"
    log "Stream profiler installed (enable: VLLM_STREAM_PROFILE=1)"
fi

# ----------------------------------------------------------
# 6. Summary
# ----------------------------------------------------------
echo ""
echo "================================================"
log "Done! Applied: $APPLIED, Already applied: $SKIPPED"
echo ""
echo "  Chat template: $TEMPLATE_DIR/chat_template.jinja"
echo ""
echo "  Example:"
echo "    vllm serve openai/gpt-oss-120b \\"
echo "      --tensor-parallel-size 2 \\"
echo "      --chat-template '$TEMPLATE_DIR/chat_template.jinja' \\"
echo "      --enable-auto-tool-choice \\"
echo "      --tool-call-parser openai"
echo "================================================"
