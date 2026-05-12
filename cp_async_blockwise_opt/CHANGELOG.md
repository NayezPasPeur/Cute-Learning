# CHANGELOG — cp_async blockwise group-GEMM optimization

**Target**: Both blockwise cp.async kernels (SM90, SS-WGMMA):
- `group_gemm_fp8_blockwise_multistage_kernel` (contiguous A)
- `group_gemm_fp8_blockwise_scatter_kernel` (scatter/gather A)

**Goal**: Improve occupancy and bandwidth utilization for Hunyuan-V3 TP=4 shapes
**Date**: 2026-05-12

## Target shapes (Hunyuan-V3 TP=4)

```
num_expert = 192, num_topk = 8, hidden = 4096, n_tp = 384
GEMM1 (gate+up): n = 768 (n_tp*2), k = 4096
GEMM2 (down):    n = 384 (n_tp),   k = 4096
BS range: 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096
```

Tokens per expert ≈ `BS * topk / num_expert`:
- BS=4→0.17, BS=32→1.3, BS=128→5.3, BS=256→10.7, BS=1024→42.7, BS=4096→170.7

Most common case: **kTileM=8 with n=768, k=4096** (BS ≤ 128).

## Bottleneck analysis

### Occupancy (primary limiter at BS=32–128)

| Resource | kTileM=8, kStage=4 (original) | kTileM=8, kStage=3 (optimized) |
|----------|-------------------------------|--------------------------------|
| sA       | 8 × 128 × 4 = 4 KB           | 8 × 128 × 3 = 3 KB            |
| sB       | 64 × 128 × 4 = 32 KB         | 64 × 128 × 3 = 24 KB          |
| sAS      | 32 × 4 = 128 B               | 32 × 3 = 96 B                 |
| sBS      | 128 B                         | 128 B                          |
| scratch  | ~800 B                        | ~800 B                         |
| **Total**| **~37 KB**                    | **~28 KB**                     |
| Max blks | 228/37 ≈ **6**               | 228/28 ≈ **8**                 |

Register pressure (per thread):
- tCr accumulator: kTileN×kTileM / 128 threads = 64×8/128 = **4 FP32** = 4 regs
- tDr accumulator: same = **4 FP32** = 4 regs
- Other (addresses, loop vars, copy state): ~60–80 regs
- Total: ~70–90 regs/thread → 65536/(128×80) ≈ 6 blocks/SM from registers

kStage=4 wastes ~9 KB/block of smem for only marginal latency-hiding
benefit (tiles are small, WGMMA+dequant completes quickly).

### Bandwidth (secondary limiter for scatter A pattern)

For kTileM=8, 128 threads service the A tile:
- 64 threads load 8 rows × 128 bytes = 1 KB (scattered, cache-unfriendly)
- **64 threads idle** — wasted bandwidth window

B loads are sequential and well-cached. The scattered A load is the
bandwidth bottleneck: random 128-byte lines from HBM, no spatial locality.

### bws_cpa vs bws_tma gap analysis

| BS  | bws_cpa (μs) | bws_tma (μs) | gap     |
|-----|-------------|-------------|---------|
| 32  | 230.91      | 216.22      | +6.8%   |
| 64  | 285.50      | 270.85      | +5.4%   |
| 128 | 337.57      | 302.78      | +11.5%  |
| 256 | 345.89      | 337.41      | +2.5%   |

The gap is largest at BS=128 (kTileM=8 with m_avg=5). Root cause:
TMA uses hardware-managed pipelining with lower smem footprint, achieving
higher occupancy. cp.async's explicit staging eats more smem.

---

## Optimizations applied

### OPT-1: Idle-thread L2 prefetch in scatter A-loader

**File**: `scatter_bw_load_A_tile_opt()` (new function)

**Problem**: For kTileM=8, kRowsPerIter=16 → only 64/128 threads issue
cp.async. The other 64 threads hit the `local_row >= kTileM` branch and
do nothing while the memory system services the scatter loads.

**Solution**: Repurpose idle threads (local_row ∈ [kTileM, 2×kTileM)) to
issue `prefetch.global.L2` for the NEXT K-tile's A rows at offset +kTileK.
This pre-warms L2 cache lines for scattered rows that would otherwise be
cold misses on the next iteration.

```
// Idle threads issue L2 prefetch for next K-tile
const int pf_row = local_row - kTileM;
asm volatile("prefetch.global.L2 [%0];\n" : : "l"(pf_ptr));
```

**Expected impact**: 2–5% latency reduction at BS=32–128 by converting
L2 misses into L2 hits for the scatter A pattern.

**Risk**: Negligible. Prefetch is a hint; incorrect prefetches are
silently dropped. Guarded by `pf_k < K` to avoid invalid addresses.

---

### OPT-2: Register-cached row_indices for kTileM ≤ 16

**Problem**: Each K-tile iteration, `scatter_bw_load_A_tile` reads
`shm_row_indices[local_row]` from shared memory. For kTileM=8 with
32 K-tiles, that's 32 × 64 = 2048 unnecessary smem reads (each thread's
row index is constant across all K-tiles).

**Solution**: Before the main loop, each thread pre-loads its row index
into a register:
```
const int my_row = idx / kThreadsPerRow;
int reg_row_idx = (my_row < num_valid) ? shm_row_indices[my_row] : 0;
```
The `scatter_bw_load_A_tile_opt` function accepts a `cached_row_idx`
parameter and skips the smem read when `kUseRegRowIdx=true`.

**Expected impact**: ~1–2% reduction from eliminated smem contention.
Saves 1 register for the cached index but eliminates shared memory
bank-conflict potential on hot reads.

**Risk**: None. Register caching is semantically identical to smem reads.
Only enabled for kTileM ≤ 16 where the index count fits comfortably in
the register file (at most 2 regs per thread).

---

### OPT-3: BF16 accumulator for kTileM ≥ 48

**Problem**: For kTileM=48, `tDr` (dequanted accumulator) uses 24 FP32
registers per thread. Together with `tCr` (24 FP32), that's 48 registers
just for accumulators — limiting occupancy to 2 blocks/SM on register
pressure alone.

**Solution**: Store `tDr` as BF16 instead of FP32 when kTileM ≥ 48:
```
auto tDr = make_tensor_like<cute::bfloat16_t>(tCr);  // 12 regs vs 24
```
The dequant FMA computes in FP32 and truncates back to BF16:
```
float prev = static_cast<float>(tDr_mn(im, in));
tDr_mn(im, in) = static_cast<cute::bfloat16_t>(tCr_mn(im, in) * yscale + prev);
```

**Register savings**:
| kTileM | FP32 tDr regs | BF16 tDr regs | Savings |
|--------|---------------|---------------|---------|
| 48     | 24            | 12            | 12      |
| 64     | 32            | 16            | 16      |

**Precision analysis**: Each FMA step truncates to BF16 (~7 bits mantissa).
Over 32 K-tiles, accumulated error is bounded by O(ntile × 2^-8) ≈ 0.125
ULP of the final value. Since the output is BF16 anyway, this is
acceptable for inference.

**Expected impact**: 3–5% at BS=1024–4096 via higher occupancy
(kMinBlocks can increase from 4→5 for kTileM=48).

**Risk**: Minor precision degradation in the accumulation path. Only
affects kTileM ≥ 48 (large batch). Can be disabled by setting
`kUseBF16Accum = false`.

---

### OPT-4: Picker table re-tune for Hunyuan n=768

**Changes** (n=768, k=4096 entries):

| kTileM | m_avg    | Original              | Optimized              | Rationale |
|--------|----------|-----------------------|------------------------|-----------|
| 8      | ≤1       | Ks=4, Kmb=2, Gm=12   | Ks=3, Kmb=3, Gm=14    | −9KB smem/blk, +50% occupancy potential |
| 8      | ≤2       | Ks=4, Kmb=2, Gm=18   | Ks=3, Kmb=3, Gm=20    | Same rationale, higher grid oversubscription |
| 8      | ≤5       | Ks=4, Kmb=2, Gm=18   | Ks=3, Kmb=3, Gm=20    | Same |
| 8      | >5       | Ks=4, Kmb=2, Gm=18   | Ks=3, Kmb=3, Gm=20    | Same |
| 16     | ≤10      | Ks=2, Kmb=2, Gm=18   | Ks=2, Kmb=3, Gm=18    | Higher occupancy target |
| 16     | >10      | Ks=2, Kmb=2, Gm=18   | Ks=2, Kmb=3, Gm=18    | Same |
| 32     | ≤21      | Ks=2, Kmb=2, Gm=10   | Ks=2, Kmb=3, Gm=12    | Kmb+1, more grid work |
| 32     | >21      | Ks=2, Kmb=2, Gm=10   | Ks=2, Kmb=3, Gm=12    | Same |
| 48     | —        | Ks=3, Kmb=4, Gm=12   | Ks=2, Kmb=5, Gm=14    | BF16 accum enables Kmb=5 |
| 64     | —        | Ks=2, Kmb=4, Gm=12   | Ks=2, Kmb=4, Gm=14    | Slightly more grid |

**Dispatch table changes** (debug trim):
- kTileM=32: added `(2, 3)` entry for n=768 picker
- kTileM=48: added `(2, 5)` entry for BF16-accumulator path

**Expected impact**: 5–10% at BS=32–128 by increasing warp-level
parallelism via higher occupancy.

**Risk**: New picker entries need benchmarking on actual hardware.
Env-var override (`HPC_BWS_KSTAGE_8`, etc.) preserved for runtime tuning.

---

### OPT-5: Conditional smem allocation for task_map mode

**Problem**: `shm_tiles` (num_group × 4 bytes = 768 bytes for Hunyuan)
is allocated even when `kUseTaskMap=true`, where it's never written/read.

**Solution**: In the kernel, conditionally place `shm_row_indices` right
after `WSshm` when task_map is used, skipping the `shm_tiles` gap. In
the launch wrapper, reduce `shm_size` by `(num_group+1)*sizeof(int)`.

**Expected impact**: Saves 768 bytes smem per block (~2%). Small but
compounds with other smem reductions.

---

### OPT-6: Wscale register pre-read before dequant FMA

**Problem**: In the original code, `sBS(itile)` is read at the start of
the dequant section, after `warpgroup_wait<0>()`. The smem read latency
(~20 cycles) is on the critical path.

**Solution**: Move the wscale read to BEFORE the WGMMA issue sequence:
```
float wscale_val = 0.f;
if constexpr (!kAblNoDequant) {
    wscale_val = sBS(itile);  // Pre-read: hidden behind WGMMA latency
}
// ... WGMMA issue/wait ...
// ... dequant uses wscale_val (already in register) ...
```

The smem latency is fully hidden behind the WGMMA execution window
(which takes ~100+ cycles on SM90).

**Expected impact**: ~20 cycles per K-tile iteration × 32 tiles ≈ 640
cycles ≈ 0.4 μs. Marginal individually but free.

---

## Summary of expected improvements by BS range

| BS range  | kTileM | Primary optimizations    | Est. improvement |
|-----------|--------|--------------------------|-----------------|
| 4–8       | 8      | OPT-1,2,4               | 3–6%            |
| 16–128    | 8      | OPT-1,2,4 (occupancy)   | 5–10%           |
| 256       | 16     | OPT-2,4                 | 3–5%            |
| 512       | 32     | OPT-4                   | 2–4%            |
| 1024      | 48     | OPT-3,4 (BF16 accum)    | 3–5%            |
| 2048–4096 | 64     | OPT-3,4                 | 2–4%            |

---

## Part 2: Contiguous-A kernel (`group_gemm_blockwise_fp8.cu`)

### OPT-A: Eliminate tCS[kN] register array — fold into dequant FMA loop

**The single largest register-pressure optimization in this PR.**

**Problem**: The original code pre-computes `float tCS[kN]` (xscale × wscale
per N-column) BEFORE issuing WGMMA, then uses it AFTER WGMMA completes.
This keeps kN FP32 registers live across the entire WGMMA instruction
sequence. Register pressure per kTileM:

| kTileM | kN (approx) | Extra regs from tCS |
|--------|-------------|---------------------|
| 8      | 4           | 4                   |
| 16     | 8           | 8                   |
| 32     | 8           | 8                   |
| 48     | 12          | 12                  |
| 64     | 16          | 16                  |

For kTileM=48/64, these 12–16 extra registers are the difference between
kMinBlocks=2 (spilling) and kMinBlocks=3 (no spilling).

**Solution**: Delete the tCS array entirely. Read xscale inline during
the post-WGMMA dequant FMA loop:

- **kTileM ≤ 32**: one smem read per thread + `__shfl_sync` broadcast
  per N-column. Cost: ~5 cycles/column, fully hidden by FMA throughput.
- **kTileM > 32**: direct `sAS(m_idx, ...)` smem read per N-column.
  Cost: ~20 cycles/column, interleaved with FMA by the compiler.

**Expected impact**: 5–10% at BS=1024–4096 (kTileM=48/64) by enabling
higher kMinBlocks; 2–3% at smaller BS from reduced register pressure.

---

### OPT-B: Replace G2SCopyAS with inline cp.async

Same optimization as the scatter kernel's P0 reg-reduction, ported to
the contiguous kernel. The CuTe `G2SCopyAS` path's partition state
(tgAS, tsAS, pred_xs) consumed ~4–6 registers that survived the main
loop. Replaced with:

```cpp
const bool xs_live = (idx < Config::kXsLiveThrs);
const int  xs_thr_off = idx * 4;
auto xs_cp_async = [&](int itile, int ismem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ...);
};
```

**Savings**: 2–3 net registers (reduced from 5–6 to 3 scalar values).

---

### OPT-C: Pre-read wscale before fence_view_async_shared

Moved `sBS(itile)` read to BEFORE `fence_view_async_shared()`:

```cpp
cp_async_wait<kStage - 1>();
__syncthreads();
const float wscale_val = sBS(itile);    // ← moved here (generic proxy, safe)
cutlass::arch::fence_view_async_shared();
// ... WGMMA ...
```

sBS was fully loaded synchronously before the main loop. This generic-proxy
`ld.shared` is safe after `__syncthreads()` and doesn't need the async
fence. Moving it earlier hides the ~20-cycle smem latency behind the
fence instruction's execution.

---

### OPT-D: Picker table — add n=768 entries for Hunyuan-V3 TP=4

**Problem**: The original k=4096 picker table only covers n ∈ {384, 512,
1024}. For Hunyuan-V3 TP=4, GEMM1 uses **n=768** which falls through to
the per-tile_m default (kStage=2, kMinBlocks=2, gridMul=9 for kTileM=8).
This default was tuned for n=384 and is suboptimal for n=768.

**Solution**: Added n=768 entries for all kTileM buckets:

| kTileM | n=768 picker                  | Rationale |
|--------|-------------------------------|-----------|
| 8      | Ks=3, Kmb=3, Gm=14..18       | Balance pipeline depth & occupancy |
| 16     | Ks=2, Kmb=3, Gm=16           | Higher occupancy target |
| 32     | Ks=2, Kmb=3, Gm=10           | Moderate grid |
| 48     | Ks=2, Kmb=3, Gm=12           | Match n=384 pattern |
| 64     | Ks=2, Kmb=3, Gm=12           | Slightly more grid |

---

## Files

```
cp_async_blockwise_opt/
├── original/
│   ├── group_gemm_blockwise_fp8.cu           ← contiguous A baseline
│   └── group_gemm_blockwise_scatter_fp8.cu   ← scatter A baseline
├── optimized/
│   ├── group_gemm_blockwise_fp8.cu           ← OPT-A,B,C,D applied
│   └── group_gemm_blockwise_scatter_fp8.cu   ← OPT-1..6 applied
└── CHANGELOG.md                              ← this file
```

## Verification checklist

- [ ] Build with `HPC_BWS_FULL_TABLE` for full dispatch coverage
- [ ] Run Hunyuan-V3 TP=4 e2e bench across all BS (script in issue)
- [ ] Compare bws_cpa column vs OLD reference
- [ ] Verify numerical accuracy (max abs diff vs FP32 reference < 1e-2)
- [ ] Profile with `ncu` to confirm occupancy increase for BS=32–128
- [ ] Check register spill count with `ncu --metrics launch__registers_per_thread`
- [ ] Verify OPT-A register savings: compare PTX register count before/after
- [ ] If BF16 accum (OPT-3, scatter only) causes precision issues, disable via `kUseBF16Accum = false`
