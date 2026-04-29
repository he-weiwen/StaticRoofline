# Memory-op coverage audit

**Status:** audit complete, no fixes applied yet. Pass at `lib/NVPTXArithIntensityPass.cpp` works correctly for ordinary global/shared/local/const/param loads and stores. Several systematic gaps and one current bug identified below.

## Problem

The pass buckets bytes by walking `MachineInstr::memoperands()` and switching on `MachineMemOperand::getAddrSpace()` (see `recordMemory` at `lib/NVPTXArithIntensityPass.cpp:127`). That implicitly assumes (a) every memory-touching instruction carries an MMO and (b) `mayLoad`/`mayStore` cleanly partition load vs. store. Both assumptions break for several NVPTX opcode families.

## Current state of the pass

Handles correctly:
- Ordinary loads/stores in addrspaces 1 (global), 3 (shared), 4 (const), 5 (local), 7 (shared-cluster), 101 (entry param). Verified against `build/test/fma_v2half.mir`, `build/test/local.mir`.
- FLOPs from scalar `FADD/FSUB/FMUL/FMA` plus packed `f16x2`/`bf16x2`/`f32x2` lane multiplier, classified per-precision, asserted PerThread scope.

Known incorrect:
- **Atomics double-bucket.** `ATOM_*`/`RED_*` set both `mayLoad=1` and `mayStore=1` (NVPTXIntrinsics.td:2446, 2475, 2502, 2713). One MMO of size N triggers `addLoadStoreBytes` (line 119) to add N to both `GlobalLoadBytes` and `GlobalStoreBytes` → 2N counted as ordinary load+store traffic. The 2N count itself is defensible as RMW requested traffic, but mixing it into ordinary load/store buckets is wrong.

Known systematic gaps (audit verified):
- LDG / LDU global loads have no MMOs at all and `mayLoad=false`.
- Generic addrspace 0 falls into `unknown_bytes`.
- cp.async, cp.async.bulk, TMA, ldmatrix, stmatrix, WMMA load/store, tcgen05_* all skip `setNodeMemRefs` in their isel path.
- Warp-cooperative ops (ldmatrix, stmatrix, WMMA, TMA bulk) report per-thread fragments; per-warp memory traffic is 32× larger.

## Verified facts (with citations)

### LDG / LDU intentionally lack MMOs

- `tryLDG()` at `llvm-project/llvm/lib/Target/NVPTX/NVPTXISelDAGToDAG.cpp:1266-1329`: builds `MachineSDNode` via `getMachineNode()` then `ReplaceNode()`. **No `setNodeMemRefs` call.**
- `tryLDU()` at `:1342-1381`: same pattern.
- Compare with the regular `tryLoad()` path at `:1176`, which **does** call `setNodeMemRefs`.
- Comments at `NVPTXIntrinsics.td:2746` (LDU) and `:2786` (LDG) confirm this is deliberate — these opcodes are not annotated `mayLoad` because they go through the read-only texture cache.

Consequence: `MI.mayLoad()` returns `false` for LDG/LDU. Detection requires opcode-name prefix matching, not MMO-presence checks.

### LDG/LDU width is operand-encoded, not just suffix-encoded

`LDG_G<NVPTXRegClass regclass>` at `NVPTXIntrinsics.td:2790` defines:
```
NVPTXInst<(outs regclass:$result),
          (ins AtomicCode:$Sign, i32imm:$fromWidth,
               UsedBytesMask:$usedBytes, ADDR:$src), ...>
```

The suffix in `LD_GLOBAL_NC_i32` names the *destination register class* (B32). The actual load width comes from the `fromWidth` immediate operand — a 32-bit destination can hold an 8/16/32-bit source with extension. Byte recovery must read MachineOperand 1 as an immediate, not assume `i32 = 4`.

Defined opcodes (NVPTXIntrinsics.td:2755-2779 LDU, :2797-2836 LDG):

| Pattern | Register class | Lane count |
|---|---|---|
| `LD_GLOBAL_NC_i{16,32,64}` / `LDU_GLOBAL_i{16,32,64}` | B16/B32/B64 | 1 |
| `LD_GLOBAL_NC_v2{i16,i32,i64}` / `LDU_GLOBAL_v2*` | same | 2 |
| `LD_GLOBAL_NC_v4{i16,i32,i64}` / `LDU_GLOBAL_v4{i16,i32}` | same | 4 |
| `LD_GLOBAL_NC_v8i32` | B32 | 8 |

Bytes = `(fromWidth / 8) * lanes`. (FIXME at `:2827` notes 8-bit LDG is broken; treat narrow forms defensively.)

### Address spaces

From `llvm-project/llvm/include/llvm/Support/NVPTXAddrSpace.h:21-31`:

| Constant | Value | NVPTX use | In our switch? |
|---|---|---|---|
| `ADDRESS_SPACE_GENERIC` | 0 | post-isel generic ptrs | **no** (→ unknown_bytes) |
| `ADDRESS_SPACE_GLOBAL` | 1 | HBM | yes |
| `ADDRESS_SPACE_SHARED` | 3 | per-CTA shared | yes |
| `ADDRESS_SPACE_CONST` | 4 | constant bank | yes |
| `ADDRESS_SPACE_LOCAL` | 5 | per-thread spill | yes |
| `ADDRESS_SPACE_TENSOR` | 6 | Blackwell tcgen05 tensor mem | **no** |
| `ADDRESS_SPACE_SHARED_CLUSTER` | 7 | Hopper DSMEM | yes |
| `ADDRESS_SPACE_ENTRY_PARAM` | 101 | byval kernel params | yes |

Note: adding `ADDRESS_SPACE_TENSOR` to the switch is **not sufficient** for tcgen05 coverage — those ops bypass MMO attachment entirely (see below).

### No-MMO opcode families

All of these skip `setNodeMemRefs` in their isel path. Static byte recovery requires an opcode classifier independent of MMOs.

| Family | Lowering site | Size source | Scope |
|---|---|---|---|
| `LD_GLOBAL_NC_*` (LDG) | `NVPTXISelDAGToDAG.cpp:1266` | suffix + `fromWidth` operand | per-thread |
| `LDU_GLOBAL_*` (LDU) | `NVPTXISelDAGToDAG.cpp:1342` | suffix + `fromWidth` operand | per-thread |
| `CP_ASYNC_{CA,CG}_SHARED_GLOBAL_{4,8,16}` | `NVPTXIntrinsics.td:482` | opcode name suffix | per-thread |
| `CP_ASYNC_BULK_*` | `NVPTXIntrinsics.td:547`, ISel `:1965` | `B32:$size` operand | warp/CTA |
| `CP_ASYNC_BULK_TENSOR_*` (TMA) | `NVPTXIntrinsics.td:2869` | tensor-map descriptor (opaque) | warp/CTA |
| `LDMATRIX` / `STMATRIX` | `NVPTXIntrinsics.td:5566/5601` | geometry in name; per-warp | warp |
| `WMMA_LOAD` / `WMMA_STORE_D` | `NVPTXIntrinsics.td:5160/5198` | fragment shape; per-warp | warp |
| `tcgen05_*` (Blackwell) | `NVPTXISelDAGToDAG.cpp:286` | intrinsic-encoded | warp/CTA |

### Atomics

`ATOM_*` and `RED_*` get MMOs with both `MOLoad | MOStore` flags (NVPTXISelLowering.cpp:4729). Definitions at NVPTXIntrinsics.td:2446, 2475, 2502, 2713 set `mayLoad=1, mayStore=1`. Detection via `MMO->isAtomic()` (preferred over `MI.isAtomic()` since the latter just delegates and returns false on no-MMO cases that can't actually be atomic anyway, but the MMO method survives both forms).

### Pseudo-memory ops (likely safe)

MBARRIER, MEMBAR, BARRIER_CTA_SYNC, CP_ASYNC_COMMIT_GROUP, CP_ASYNC_WAIT_* use `BasicNVPTXInst` without `mayLoad`/`mayStore` declarations (NVPTXIntrinsics.td:1024-1098, :363-405, :499-507). Should not pollute byte counts. **Unverified:** whether intrinsic `MemoryEffects` in IntrinsicsNVVM.td cause isel to attach MMOs anyway. Needs ground-truth experiment.

### Textures / surfaces

`TEX_*` opcodes exist as real instructions (defined via `defm` macros at NVPTXIntrinsics.td:3015+, not inline asm — earlier audit incorrectly reported these missing). Surface ops likely similar. MMO behavior unverified. Low priority for AI workloads; uses texref/sampler descriptors so `getAddrSpace()` would not be meaningful even if MMOs are attached.

## Action items (priority order)

### 1. Atomic buckets

Detect via `MMO->isAtomic()`. Route to dedicated buckets, **not** the load/store path. Two counters per addrspace:
- `atomic_operand_bytes` = data width (4 B for u32 atomic.add). Use this if AI denominator wants minimum requested traffic.
- `atomic_rmw_estimated_bytes` = 2 × width. Use this for cache-traffic-style roofline.

Split by addrspace: global atomics serialize on memory controllers; shared atomics are single-clock post-Volta. `global_atomic_*` and `shared_atomic_*` are meaningful different signals.

Make explicit which counter feeds `ai`. Add a regression test (no atomic test exists yet).

### 2. LDG / LDU fallback

When opcode name starts with `LD_GLOBAL_NC_` or `LDU_GLOBAL_`:
- Lane count from suffix prefix (`v2` / `v4` / `v8` / scalar).
- Per-lane width from MachineOperand 1 (`fromWidth` immediate, in bits). Defensive: skip if not an immediate.
- Bucket as `global_load_bytes` (always addrspace 1 for these families).

Add a test using `__ldg(...)` or `__restrict__` global pointers to exercise the path.

### 3. Generic addrspace + no-MMO diagnostic

- Add `generic_load_bytes` / `generic_store_bytes` for addrspace 0. Do **not** silently alias to global — generics that survive to MIR are precisely the cases the compiler couldn't prove. Optionally add `generic_assumed_global_bytes` as a separate user-controlled policy knob.
- Add a counter for "memory-looking instruction with no MMO" (`mayLoad || mayStore` true, `memoperands().empty()`, AND not in our opcode-classifier whitelist). Surfaces future blind spots automatically.

### 4. Simple cp.async fallback

`CP_ASYNC_{CA,CG}_SHARED_GLOBAL_{4,8,16}` is per-thread, byte size in opcode name. No scope-design dependency. Implement now.

Two-sided traffic in one instruction: count both `global_load_bytes` (the source side, AI-relevant) and `shared_store_bytes` (the destination, diagnostic only — should not enter AI denominator unless explicitly requested).

### 5. Ground-truth investigations

- Compile a kernel using `int_nvvm_mbarrier_*` intrinsics. Inspect MIR. Confirm no `:: (load …)` annotations on MBARRIER instructions. If any appear, those need explicit exclusion.
- Compile a kernel using `int_nvvm_tex_*`. Inspect MIR for MMOs on `TEX_*` ops. Determine bucketing.

### 6. Memory-side `InvocationScope` design

Required before implementing ldmatrix, stmatrix, WMMA, TMA bulk, tcgen05. Mirrors the FLOP-side scope work. Per-warp byte counts must aggregate as `bytes × 1` per warp instance (already correct for the `bb.0` view, but matters when computing per-thread or per-CTA derived quantities). Decide:
- whether per-instruction reporting is per-thread or per-warp,
- whether the AI denominator multiplies warp-scoped bytes by anything,
- how to report mixed PerThread + PerWarp memory in one block.

## Out of scope at this layer

- **Inline PTX asm bodies.** Already gated specially at `lib/NVPTXArithIntensityPass.cpp:266`. The PTX/Classifier path is the eventual fix; this is independent of the memory-op audit.
- **ptxas-inserted spills, SASS-only side effects.** Happen after LLVM lowers to PTX. Not addressable without profiler/SASS integration.
- **TMA descriptor contents.** Tensor-map descriptors are opaque at MIR level. Recoverable only with IR-level analysis or user-supplied metadata.

## Reference

Pass: `lib/NVPTXArithIntensityPass.cpp`. Memory bucketing in `recordMemory` (line 127), load/store split in `addLoadStoreBytes` (line 119), AI computation in `printFlopsAndBytes` (line 188).

LLVM tree: `/home/whe302/compilers/llvm-project/llvm/lib/Target/NVPTX/`. Key files for this work: `NVPTXISelDAGToDAG.cpp`, `NVPTXIntrinsics.td`, `NVPTXInstrInfo.td`, `NVPTXISelLowering.cpp`. Address-space constants: `llvm-project/llvm/include/llvm/Support/NVPTXAddrSpace.h`.
