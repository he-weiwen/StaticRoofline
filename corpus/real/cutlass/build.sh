#!/usr/bin/env bash
# Compile a curated set of CUTLASS examples to MIR and run the ptx-ai analyzer
# on each. Reports compile success/failure, kernel count, and the histogram of
# unrecognized opcodes so we can drive coverage closure.
#
# Configuration via environment:
#   CUTLASS_PATH    : path to cutlass clone (default: $HOME/compilers/cutlass)
#   CUDA_PATH       : path to CUDA toolkit (default: /usr/local/cuda-13.0)
#   LLVM_PATH       : path to local LLVM build (default: $HOME/compilers/llvm-project/build)
#   ANALYZER_LIB    : path to NVPTXArithIntensity.so (default: ../../../build/lib/...)
#   OUT_DIR         : where to put generated .ll/.mir/.log (default: ./build)

set -uo pipefail

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

CUTLASS_PATH="${CUTLASS_PATH:-$HOME/compilers/cutlass}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda-13.0}"
LLVM_PATH="${LLVM_PATH:-$HOME/compilers/llvm-project/build}"
ANALYZER_LIB="${ANALYZER_LIB:-$HERE/../../../build/lib/NVPTXArithIntensity.so}"
OUT_DIR="${OUT_DIR:-$HERE/build}"

CLANG="$LLVM_PATH/bin/clang++"
LLC="$LLVM_PATH/bin/llc"

# Sanity checks
for tool in "$CLANG" "$LLC" "$ANALYZER_LIB"; do
  [ -e "$tool" ] || { echo "ERROR: missing $tool" >&2; exit 1; }
done
[ -d "$CUTLASS_PATH/examples" ] || {
  echo "ERROR: CUTLASS_PATH=$CUTLASS_PATH does not contain examples/" >&2; exit 1;
}

mkdir -p "$OUT_DIR"

# Common include paths
INCLUDES=(
  -I "$CUTLASS_PATH/include"
  -I "$CUTLASS_PATH/tools/util/include"
  -I "$CUTLASS_PATH/examples/common"
  -I "$CUDA_PATH/targets/x86_64-linux/include/cccl"
)

# Common compile flags. Note -nocudalib is intentionally NOT set: we need the
# CUDA runtime declarations (cudaConfigureCall etc.) to be visible so that the
# host-side <<<grid,block>>> launch sites parse cleanly even though we discard
# the host code with --cuda-device-only.
CFLAGS=(
  --cuda-device-only
  --cuda-path="$CUDA_PATH"
  -emit-llvm -S -O3
  -ffp-contract=fast
  -std=c++17
  -Wno-unknown-cuda-version
  -Wno-cuda-compat                   # silence Hopper #pragma unroll(...) noise
  # NVCC defines these; clang doesn't. Many CUTLASS examples gate features
  # on them. We claim CUDA 12.7 — recent enough to enable Hopper/Blackwell
  # codepaths but BELOW the 12.8 threshold that activates __nv_atomic_load_n
  # (an NVCC-only intrinsic clang can't compile).
  -D__CUDACC_VER_MAJOR__=12
  -D__CUDACC_VER_MINOR__=7
  -D__CUDACC_VER_BUILD__=0
  -D__CUDACC_VER__=120700
  # CUDA 13 versioned these typedefs; CUTLASS still uses the unversioned
  # names. Redirect via macro so cutlass/cuda_host_adapter.hpp typechecks.
  -DPFN_cuTensorMapEncodeTiled=PFN_cuTensorMapEncodeTiled_v12000
  -DPFN_cuTensorMapEncodeIm2col=PFN_cuTensorMapEncodeIm2col_v12000
)

compile_to_mir() {
  local example_dir="$1" source_file="$2" arch="$3" stem="$4"
  local src="$CUTLASS_PATH/examples/$example_dir/$source_file"
  local ll="$OUT_DIR/$stem.ll"
  local mir="$OUT_DIR/$stem.mir"
  local log="$OUT_DIR/$stem.compile.log"
  local report="$OUT_DIR/$stem.report.txt"

  if [ ! -f "$src" ]; then
    echo "MISS  $stem  ($src not found)"
    return 1
  fi

  # Stage 1: clang CUDA → LLVM IR
  if ! "$CLANG" "${CFLAGS[@]}" --cuda-gpu-arch="$arch" "${INCLUDES[@]}" \
        "$src" -o "$ll" >"$log" 2>&1; then
    echo "FAIL  $stem  ($arch, clang) — see $(basename "$log")"
    return 1
  fi

  # Stage 2: llc IR → MIR
  if ! "$LLC" -march=nvptx64 -mcpu="$arch" -O3 \
        -stop-before=nvptx-asm-printer \
        "$ll" -o "$mir" >>"$log" 2>&1; then
    echo "FAIL  $stem  ($arch, llc) — see $(basename "$log")"
    return 1
  fi

  # Stage 3: analyzer pass
  if ! "$LLC" -load "$ANALYZER_LIB" -run-pass=ptx-ai \
        "$mir" -o /dev/null >"$report" 2>&1; then
    echo "FAIL  $stem  ($arch, analyzer) — see $(basename "$report")"
    return 1
  fi

  # Quick metrics from the report
  local kernels
  kernels=$(grep -c "^kernel " "$report" || true)
  if [ "$kernels" -eq 0 ]; then
    echo "EMPTY $stem  ($arch)  — compiled cleanly but produced no kernels"
    echo "      (typical cause: arch-specific templates failed to instantiate at this --cuda-gpu-arch)"
    return 2   # distinguish from FAIL so it doesn't count as a hard failure
  fi
  echo "PASS  $stem  ($arch)  kernels=$kernels"
  return 0
}

OK=0
FAIL=0
EMPTY=0
TARGETS_FILE="$HERE/targets.txt"

while IFS=: read -r ex_dir src_file arch comment; do
  # Skip blank lines and comments
  [[ -z "$ex_dir" || "$ex_dir" == \#* ]] && continue
  stem=$(echo "$ex_dir" | tr -c 'a-zA-Z0-9' '_' | sed 's/_$//')

  compile_to_mir "$ex_dir" "$src_file" "$arch" "$stem"
  case $? in
    0) OK=$((OK+1)) ;;
    2) EMPTY=$((EMPTY+1)) ;;
    *) FAIL=$((FAIL+1)) ;;
  esac
done < "$TARGETS_FILE"

echo
echo "===================================="
echo "Summary: $OK passed, $EMPTY empty, $FAIL failed"
echo "Outputs: $OUT_DIR/"
echo "===================================="

# Top unrecognized opcodes across the whole corpus (anything not in the
# explicit ignore-set we trust). Because the analyzer doesn't yet emit a
# dedicated unknown bucket, we approximate by extracting opcode histograms
# and filtering known categories.
echo
echo "=== Aggregate opcode histogram (top 30 by frequency) ==="
cat "$OUT_DIR"/*.report.txt 2>/dev/null \
  | grep -E "^    [A-Z][A-Za-z0-9_]*: [0-9]+$" \
  | sed -E 's/^    ([^:]+): ([0-9]+)$/\2 \1/' \
  | awk '{count[$2] += $1} END {for (op in count) print count[op], op}' \
  | sort -rn | head -30
