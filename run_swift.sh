#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
MODEL=${P36_SWIFT_MODEL:-"$ROOT/gguf/Swift-Qwen3.8-27B-IQ3_XXS.gguf"}
if [ -n "${P36_SWIFT_CTX:-}" ]; then
    CTX=$P36_SWIFT_CTX
else
    case "$MODEL" in
        *IQ4_XS.gguf) CTX=1024 ;;
        *) CTX=4096 ;;
    esac
fi
DRAFT=${P36_SWIFT_MTP_DRAFT:-1}
MARGIN=${P36_SWIFT_MTP_MARGIN:-3}

if [ ! -s "$MODEL" ]; then
    echo "Swift model not found: $MODEL" >&2
    echo "Download it with: $ROOT/download_model.sh swift" >&2
    exit 1
fi

if [ ! -x "$ROOT/p36" ]; then
    echo "p36 is not built; run: make vulkan-generic" >&2
    exit 1
fi

# These are safe, quality-preserving paths for the BC-250. The runtime already
# selects them by default, but setting them here makes this launcher stable
# across shell environments and future default changes.
export P36_VK_DENSE_IQ3_FAST=${P36_VK_DENSE_IQ3_FAST:-1}
export P36_VK_DENSE_KQUANT_DECODE=${P36_VK_DENSE_KQUANT_DECODE:-1}
export P36_VK_DENSE_KQUANT_MMQ=${P36_VK_DENSE_KQUANT_MMQ:-1}
export P36_VK_GPU_ATTN_POST=${P36_VK_GPU_ATTN_POST:-1}
export P36_VK_GPU_RMS=${P36_VK_GPU_RMS:-1}
export P36_VK_GPU_FFN_TAIL=${P36_VK_GPU_FFN_TAIL:-1}
export P36_VK_GPU_SWIGLU=${P36_VK_GPU_SWIGLU:-1}
export P36_VK_DENSE_WEIGHT_COPY_PARALLEL=${P36_VK_DENSE_WEIGHT_COPY_PARALLEL:-1}

# IQ4_XS is a 14.6 GiB dense model. Keep a 7 GiB prefix and recycle a 2 GiB
# working set; this is the safe profile for the BC-250's 10.2 GiB device-local
# heap. Full resident prewarm is intentionally disabled because it can trigger
# the host OOM killer even when Vulkan reports enough aggregate UMA memory.
case "$MODEL" in
    *IQ4_XS.gguf)
        export P36_VK_PREWARM=${P36_VK_PREWARM:-0}
        export P36_VK_DENSE_WEIGHT_CACHE_GIB=${P36_VK_DENSE_WEIGHT_CACHE_GIB:-9}
        export P36_VK_DENSE_WEIGHT_PIN_GIB=${P36_VK_DENSE_WEIGHT_PIN_GIB:-7}
        ;;
esac

exec "$ROOT/p36" --vulkan -m "$MODEL" --ctx "$CTX" \
    --mtp-draft "$DRAFT" --mtp-margin "$MARGIN" "$@"
