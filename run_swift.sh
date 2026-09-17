#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
MODEL=${Q36_SWIFT_MODEL:-"$ROOT/gguf/Swift-Qwen3.8-27B-IQ4_XS.gguf"}
if [ -n "${Q36_SWIFT_CTX:-}" ]; then
    CTX=$Q36_SWIFT_CTX
else
    case "$MODEL" in
        *IQ4_XS.gguf) CTX=1024 ;;
        *) CTX=4096 ;;
    esac
fi
DRAFT=${Q36_SWIFT_MTP_DRAFT:-3}
MARGIN=${Q36_SWIFT_MTP_MARGIN:-3}

if [ ! -s "$MODEL" ]; then
    echo "Swift model not found: $MODEL" >&2
    echo "Download it with: $ROOT/download_model.sh swift" >&2
    exit 1
fi

if [ ! -x "$ROOT/q36" ]; then
    echo "q36 is not built; run: make vulkan-generic" >&2
    exit 1
fi

# These are safe, quality-preserving paths for the BC-250. The runtime already
# selects them by default, but setting them here makes this launcher stable
# across shell environments and future default changes.
export Q36_VK_DENSE_IQ3_FAST=${Q36_VK_DENSE_IQ3_FAST:-1}
export Q36_VK_DENSE_KQUANT_DECODE=${Q36_VK_DENSE_KQUANT_DECODE:-1}
export Q36_VK_DENSE_KQUANT_MMQ=${Q36_VK_DENSE_KQUANT_MMQ:-1}
export Q36_VK_GPU_ATTN_POST=${Q36_VK_GPU_ATTN_POST:-1}
export Q36_VK_GPU_RMS=${Q36_VK_GPU_RMS:-1}
export Q36_VK_GPU_FFN_TAIL=${Q36_VK_GPU_FFN_TAIL:-1}
export Q36_VK_GPU_SWIGLU=${Q36_VK_GPU_SWIGLU:-1}

# IQ4_XS is a 14.6 GiB dense model. Do not prewarm it into the GPU arena;
# recycle a bounded 2 GiB working set instead. This trades throughput for a
# reliable run on the 16 GiB BC-250 and can be overridden for larger GPUs.
case "$MODEL" in
    *IQ4_XS.gguf)
        export Q36_VK_PREWARM=${Q36_VK_PREWARM:-0}
        export Q36_VK_DENSE_WEIGHT_CACHE_GIB=${Q36_VK_DENSE_WEIGHT_CACHE_GIB:-2}
        ;;
esac

exec "$ROOT/q36" --vulkan -m "$MODEL" --ctx "$CTX" \
    --mtp-draft "$DRAFT" --mtp-margin "$MARGIN" "$@"
