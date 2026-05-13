// Copyright 2026 hpc-ops authors
//
// Blockwise FP8 group-GEMM SCATTER variant — cp.async + SS-WGMMA (SM90).
//
// Identical to `group_gemm_fp8_blockwise_multistage_kernel` except for A
// loading: each token's row is gathered from `Aptr` via `row_indices`
// (an int32 permutation array, length = total_tokens). This matches the
// API of the non-blockwise scatter kernel
// (`group_gemm_fp8_scatter_kernel`) but with DeepSeek-style blockwise
// (xscale, wscale) dequant in place of the per-expert y_scale.
//
// Contract:
//   row_indices[start_token + t] gives the row in Aptr for the t-th
//   token of expert `igroup`, where start_token =
//   cu_seqlens_ptr[igroup]. xscale layout is identical to the non-
//   scatter blockwise kernel's contract: (num_block_k, m_pad), stride
//   (m_pad, 1). The CALLER is responsible for placing each token's
//   xscale at the m_pad slot matching its (igroup, itile_m, m_local)
//   — i.e. xscale is already "scattered" in m_pad layout and we don't
//   gather it again here.
//
// All other invariants (cross-proxy fence, wscale preload pattern,
// fence_view_async_shared, etc.) are inherited from the non-scatter
// blockwise kernel.

#include <cuda.h>
#include <stdio.h>

// ABLATION BUILD: when HPC_BWS_ABLATION_EXPAND is defined, the launch
// wrapper switches on HPC_BWS_ABLATE env var to instantiate one of 8
// ablation kernel variants (mask 0..7 over {NoLoadAS, NoLoadBS,
// NoDequant}). Off by default — only mask=0 is compiled. The kernel
// template still accepts kAblMask so the source structure is preserved.
//
// Similarly, HPC_BWS_FULL_TABLE re-enables the full picker dispatch
// table (16+ combos per kTileM). Off by default — only the combos
// hit by current debug shape (Hunyuan TP=4 BS=32) are compiled, which
// drops the per-build TU compile time from ~5min to ~1min.
#define HPC_BWS_ABLATION_BUILD 1

#include <cassert>
#include <cstdio>   // snprintf
#include <cstdlib>  // getenv, atoi

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/memory_sm80.h"
#include "src/group_gemm/sm90/cp_async/common.cuh"
#include "src/group_gemm/sm90/cp_async/config.h"
#include "src/group_gemm/sm90/cp_async/group_gemm.h"
#include "src/utils/utils.h"
#include "src/utils/utils.cuh"  // retile_fragment

namespace hpc {
namespace group_gemm_cp_async {

namespace kernels {

// Scatter A-loader: each token's K-tile is fetched via row_indices.
// Mirrors the contiguous A-load pattern from the blockwise non-scatter
// kernel, but issues 16-byte cp.async per (row, k_atom) pair using
// the row index from `shm_row_indices`.
template <typename Config, bool kFullTile, typename TsACopy>
__device__ __forceinline__ void scatter_bw_load_A_tile(
    TsACopy &tsA_copy,
    const typename Config::Tin *__restrict__ A_pool,
    const int *__restrict__ shm_row_indices,
    int num_valid, int k_col_base, int K, int ismem) {
  using namespace cute;  // NOLINT
  using Tin = typename Config::Tin;
  constexpr int kTileM = Config::kTileM;
  constexpr int kTileK = Config::kTileK;
  constexpr int kElemsPerAtom = 16;  // 16 fp8 = 128 bits
  constexpr int kThreadsPerRow = kTileK / kElemsPerAtom;
  constexpr int kRowsPerIter = 128 / kThreadsPerRow;
  constexpr int kNumIters = (kTileM + kRowsPerIter - 1) / kRowsPerIter;
  constexpr bool kHasVirtualOOB = (kNumIters * kRowsPerIter > kTileM);

  const int row_in_group = threadIdx.x / kThreadsPerRow;
  const int k_thread = threadIdx.x % kThreadsPerRow;
  const int k_col = k_col_base + k_thread * kElemsPerAtom;

  auto do_one = [&](int row_iter, int local_row) {
    void *smem_ptr = (void *)&tsA_copy(cute::Int<0>{}, row_iter,
                                       cute::Int<0>{}, ismem);
    int global_row;
    int src_size;
    if constexpr (kFullTile) {
      global_row = shm_row_indices[local_row];
      src_size = 16;
    } else {
      const bool valid = local_row < num_valid;
      global_row = valid ? shm_row_indices[local_row] : 0;
      src_size = valid ? 16 : 0;
    }
    const void *gmem_ptr =
        (const void *)&A_pool[uint64_t(global_row) * K + k_col];
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(
            static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr))),
        "l"(gmem_ptr), "n"(16), "r"(src_size));
  };

#pragma unroll
  for (int row_iter = 0; row_iter < kNumIters; ++row_iter) {
    const int local_row = row_iter * kRowsPerIter + row_in_group;
    if constexpr (kHasVirtualOOB) {
      if (local_row < kTileM) {
        do_one(row_iter, local_row);
      }
    } else {
      do_one(row_iter, local_row);
    }
  }
}

// ─── ABLATION (set via env var HPC_BWS_ABLATE = bitmask) ───
// bit 0 (1): skip per-tile cp.async load of xscale (sAS)   -> kAblNoLoadAS
// bit 1 (2): skip one-shot cp.async preload of wscale (sBS) -> kAblNoLoadBS
// bit 2 (4): skip post-WGMMA dequant FMA loop               -> kAblNoDequant
// Template params so disabled paths are fully DCE'd (no register pressure).
template <typename Config, bool kUseTaskMap, bool kUsePDL, int kMinBlocks,
          int kAblMask = 0>
__global__ void __launch_bounds__(128, kMinBlocks)
    group_gemm_fp8_blockwise_scatter_kernel(
        const void *Cptr, const void *Aptr, const void *Bptr,
        const float *xscale_ptr, const float *wscale_ptr,
        const int *row_indices_ptr,
        const int *seqlens_ptr, const int *cu_seqlens_ptr,
        const int *tiles_ptr, const int *cu_tiles_ptr,
        const int4 *task_map_ptr, int task_map_len,
        int m, int n, int k, int num_group, int m_pad,
        int num_block_n, int num_block_k_pad4,
        cutlass::FastDivmod flat_divider) {
  constexpr bool kAblNoLoadAS = (kAblMask & 1) != 0;
  constexpr bool kAblNoLoadBS = (kAblMask & 2) != 0;
  constexpr bool kAblNoDequant = (kAblMask & 4) != 0;
  using namespace cute;  // NOLINT

  using Tin = typename Config::Tin;
  using Tout = typename Config::Tout;
  using TS = typename Config::TS;

  using SmemLayoutA = typename Config::SmemLayoutA;
  using SmemLayoutB = typename Config::SmemLayoutB;
  using SmemLayoutCT = typename Config::SmemLayoutCT;
  using SmemLayoutXS = typename Config::SmemLayoutXS;
  using SmemLayoutWS = typename Config::SmemLayoutWS;

  using G2SCopyA = typename Config::G2SCopyA;
  using G2SCopyB = typename Config::G2SCopyB;
  // G2SCopyAS removed — replaced by inline-asm cp.async (see main loop).
  using S2GCopyC = typename Config::S2GCopyC;
  using TiledMMA = typename Config::TiledMMA;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kStage = Config::kStage;
  constexpr int kMaxNtileK = Config::kMaxNtileK;
  constexpr int kWScaleNDiv = Config::kWScaleNDiv;

  static_assert(kTileK == 128,
                "blockwise scatter FP8 cp.async kernel requires kTileK == 128");
  static_assert(kTileN == 64,
                "blockwise scatter FP8 cp.async kernel currently assumes kTileN == 64");

  extern __shared__ uint8_t shm_data[];
  Tin *Ashm  = reinterpret_cast<Tin  *>(shm_data);
  Tin *Bshm  = Ashm + cosize(SmemLayoutA{});
  TS  *ASshm = reinterpret_cast<TS *>(Bshm + cosize(SmemLayoutB{}));
  TS  *WSshm = ASshm + cosize(SmemLayoutXS{});
  int *shm_tiles = reinterpret_cast<int *>(WSshm + cosize(SmemLayoutWS{}));
  int *shm_row_indices = shm_tiles + num_group;
  Tout *Cshm = reinterpret_cast<Tout *>(shm_data);  // overlaps A

  int idx = threadIdx.x;
  int iblock = blockIdx.x;

  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }

  if constexpr (!kUseTaskMap) {
    for (int i = idx; i < num_group; i += blockDim.x) {
      shm_tiles[i] = tiles_ptr[i];
    }
    __syncthreads();
  }

  int ntile = k / kTileK;
  int igroup = 0;
  int itile_m, itile_n;
  int sum_tile_m = 0;

  while (true) {
    if constexpr (kUseTaskMap) {
      if (iblock >= task_map_len) break;
      int4 task = task_map_ptr[iblock];
      igroup = task.x;
      if (igroup < 0) break;
      itile_m = task.y;
      itile_n = task.z;
    } else {
      get_next_tile_horizon(shm_tiles, iblock, num_group, igroup, itile_m,
                            itile_n, sum_tile_m, flat_divider);
      if (igroup < 0) break;
    }

    int start_token = cu_seqlens_ptr[igroup];
    int m_group = seqlens_ptr[igroup];  // actual tokens this expert
    iblock += gridDim.x;

    // ───────── Global tensors (B/C) ─────────
    // A is gathered per-row via row_indices below; no global tensor.
    Tensor B = make_tensor(
        make_gmem_ptr(reinterpret_cast<const Tin *>(Bptr)
                      + uint64_t(igroup) * n * k),
        make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor C = make_tensor(
        make_gmem_ptr(reinterpret_cast<Tout *>(const_cast<void *>(Cptr))
                      + uint64_t(start_token) * n),
        make_shape(n, m_group), make_stride(Int<1>{}, n));

    Tensor gB  = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}),
                            make_coord(itile_n, _));
    Tensor gCT = local_tile(C, make_tile(Int<kTileN>{}, Int<kTileM>{}),
                            make_coord(itile_n, itile_m));

    int m_tile_base = cu_tiles_ptr[igroup] * kTileM + itile_m * kTileM;
    // (gAS tensor removed — xscale gmem ptr is computed directly in
    //  xs_gmem_addr below; see "P0 reg-reduction" block.)

    int ws_n_blk = itile_n / kWScaleNDiv;
    const float *ws_base = wscale_ptr
                           + uint64_t(igroup) * num_block_n * num_block_k_pad4
                           + uint64_t(ws_n_blk) * num_block_k_pad4;

    // ───────── Shared memory tensors ─────────
    auto sA  = make_tensor(make_smem_ptr(Ashm),  SmemLayoutA{});   // (kTileM, kTileK, kStage)
    auto sB  = make_tensor(make_smem_ptr(Bshm),  SmemLayoutB{});   // (kTileN, kTileK, kStage)
    auto sCT = make_tensor(make_smem_ptr(Cshm),  SmemLayoutCT{});  // (kTileN, kTileM)
    auto sAS = make_tensor(make_smem_ptr(ASshm), SmemLayoutXS{});  // (kTileM, 1, kStage)
    auto sBS = make_tensor(make_smem_ptr(WSshm), SmemLayoutWS{});  // (kMaxNtileK,)

    // ───────── Copy partitions ─────────
    G2SCopyA g2s_tiled_copy_a;
    auto g2s_thr_a = g2s_tiled_copy_a.get_slice(idx);
    auto tsA = g2s_thr_a.partition_D(sA);

    G2SCopyB g2s_tiled_copy_b;
    auto g2s_thr_b = g2s_tiled_copy_b.get_slice(idx);
    auto tgB = g2s_thr_b.partition_S(gB);
    auto tsB = g2s_thr_b.partition_D(sB);

    // P0 reg-reduction (2026-05-12): replace cute::TiledCopy(G2SCopyAS) with
    // hand-rolled inline asm cp.async. The CuTe path's partition_S/D state
    // (tgAS, tsAS — gmem stride/offset, smem stride/offset, copy-rank
    // metadata) ate ~4-6 registers that survived across the entire main
    // loop. We replace them with two precomputed scalar pointers:
    //
    //   xs_gmem_base[ismem]  = xscale_ptr + m_tile_base + ismem*m_pad + thr_idx*4
    //   xs_smem_base[ismem]  = ASshm + ismem*kTileM + thr_idx*4
    //
    // Across iterations we only update by `m_pad * sizeof(fp32)` for gmem
    // (precomputed once) and stride by kTileM*sizeof(fp32) for smem.
    //
    // Each live thread (idx < kXsLiveThrs) loads exactly 4 fp32 = 16B per
    // tile, identical to the original ValLayout=(4,1).
    const bool xs_live = (idx < Config::kXsLiveThrs);
    const int  xs_thr_off  = idx * 4;  // 0,4,8,...,kTileM-4
    const float *xs_gmem_t = xscale_ptr + m_tile_base + xs_thr_off;
    // smem ptr for THIS thread, ismem-indexed: ASshm + ismem*kTileM + xs_thr_off
    auto xs_smem_addr = [&](int ismem) {
      return cast_smem_ptr_to_uint(ASshm + ismem * kTileM + xs_thr_off);
    };
    auto xs_gmem_addr = [&](int itile) {
      return reinterpret_cast<const void *>(xs_gmem_t + itile * m_pad);
    };
    auto xs_cp_async = [&](int itile, int ismem) {
      // 16B cp.async; predicated by xs_live at the call site.
      asm volatile(
          "cp.async.cg.shared.global [%0], [%1], 16;\n"
          :
          : "r"(xs_smem_addr(ismem)), "l"(xs_gmem_addr(itile)));
    };

    S2GCopyC s2g_tiled_copy_c;
    auto s2g_thr_c = s2g_tiled_copy_c.get_slice(idx);
    auto tCs2g = s2g_thr_c.partition_S(sCT);
    auto tCg2s = s2g_thr_c.partition_D(gCT);

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(idx);
    auto tBr = thr_mma.make_fragment_A(thr_mma.partition_A(sB));
    auto tAr = thr_mma.make_fragment_B(thr_mma.partition_B(sA));
    auto tCr = thr_mma.partition_fragment_C(gCT);

    // Identity tensor to recover (n, m) coords per accumulator slot.
    auto gI = make_identity_tensor(gCT.shape());
    auto tI = thr_mma.partition_C(gI);
    auto tI_mn = retile_fragment(tI);
    auto tCr_mn = retile_fragment(tCr);
    constexpr int kM = size<0>(tCr_mn);
    constexpr int kN = size<1>(tCr_mn);

    // ───────── Predicate for C epilogue ─────────
    auto tIC = s2g_thr_c.partition_S(
        make_identity_tensor(make_shape(Int<kTileN>{}, Int<kTileM>{})));
    auto pred_c = make_tensor<bool>(shape(tIC));
#pragma unroll
    for (int i = 0; i < size(tIC); ++i) {
      pred_c(i) = get<1>(tIC(i)) < kTileM &&
                  itile_m * kTileM + get<1>(tIC(i)) < m_group;
    }

    // ───────── Stage row_indices into shmem (one slot per tile-row) ─────────
    const int token_base_for_tile = start_token + itile_m * kTileM;
    const int remaining = m_group - itile_m * kTileM;
    const int num_valid = remaining < kTileM ? remaining : kTileM;
    const bool is_full_tile = (num_valid == kTileM);
    for (int i = idx; i < num_valid; i += blockDim.x) {
      shm_row_indices[i] = row_indices_ptr[token_base_for_tile + i];
    }
    __syncthreads();

    // ───────── Preload wscale (one-shot, async — drains naturally) ─────────
    {
      constexpr int kWsVecs = kMaxNtileK / 4;
      if constexpr (!kAblNoLoadBS) {
        if (idx < kWsVecs) {
          int k_vec = idx * 4;
          if (k_vec < num_block_k_pad4) {
            uint32_t smem_int_ptr = cast_smem_ptr_to_uint(WSshm + k_vec);
            const float *gmem_ptr = ws_base + k_vec;
            asm volatile(
                "cp.async.cg.shared.global [%0], [%1], 16;\n"
                :
                : "r"(smem_int_ptr), "l"(gmem_ptr));
          }
        }
      }
      cp_async_fence();
    }

    // ───────── Prologue: prefetch kStage-1 stages of (A, B, XS) ─────────
    int itile_to_read = 0;
    int ismem_write = 0;
#pragma unroll
    for (int i = 0; i < kStage - 1; ++i) {
      if (i < ntile) {
        if (is_full_tile) {
          scatter_bw_load_A_tile<Config, /*kFullTile=*/true>(
              tsA, (const Tin *)Aptr, shm_row_indices, num_valid,
              i * kTileK, k, i);
        } else {
          scatter_bw_load_A_tile<Config, /*kFullTile=*/false>(
              tsA, (const Tin *)Aptr, shm_row_indices, num_valid,
              i * kTileK, k, i);
        }
        cute::copy(g2s_tiled_copy_b, tgB(_, _, _, i), tsB(_, _, _, i));
        if constexpr (!kAblNoLoadAS) {
          if (xs_live) {
            xs_cp_async(/*itile=*/i, /*ismem=*/i);
          }
        }
        // FIX 2026-05-12: increments must stay INSIDE the (i < ntile) guard,
        // otherwise short-k shapes (ntile < kStage-1) advance the counters
        // past valid stages, and the main loop will read garbage smem at
        // stages [ntile, kStage-1). Compare with non-scatter sibling
        // group_gemm_blockwise_fp8.cu L237-239.
        ++itile_to_read;
        ++ismem_write;
      }
      cp_async_fence();
    }

    // Main loop accumulator.
    auto tDr = make_tensor_like(tCr);
    clear(tDr);

    // ───────── Main loop ─────────
    for (int itile = 0; itile < ntile; ++itile) {
      if (itile_to_read < ntile) {
        if (is_full_tile) {
          scatter_bw_load_A_tile<Config, /*kFullTile=*/true>(
              tsA, (const Tin *)Aptr, shm_row_indices, num_valid,
              itile_to_read * kTileK, k, ismem_write);
        } else {
          scatter_bw_load_A_tile<Config, /*kFullTile=*/false>(
              tsA, (const Tin *)Aptr, shm_row_indices, num_valid,
              itile_to_read * kTileK, k, ismem_write);
        }
        cute::copy(g2s_tiled_copy_b,
                   tgB(_, _, _, itile_to_read),
                   tsB(_, _, _, ismem_write));
        if constexpr (!kAblNoLoadAS) {
          if (xs_live) {
            xs_cp_async(/*itile=*/itile_to_read, /*ismem=*/ismem_write);
          }
        }
        ++itile_to_read;
        ismem_write = (ismem_write + 1) % kStage;
      }
      cp_async_fence();
      
      cp_async_wait<kStage - 1>();
      __syncthreads();
      // EXPERIMENT 2026-05-12: removed cutlass::arch::fence_view_async_shared();
      // xs/ws are read by cuda-core (generic proxy) ld.shared, and they were
      // written by cp.async (also generic proxy). __syncthreads() above already
      // provides the necessary cta-wide visibility within the generic proxy.
      // The fence is only needed if a consumer in the *async* proxy (e.g.
      // wgmma) needs to see the cp.async writes — but wgmma reads A/B via its
      // own descriptor path which has its own implicit synchronization
      // (warpgroup_arrive). So this fence is suspected redundant here.

      const int ismem_read = itile % kStage;

      // Issue one WGMMA batch over the K dimension.
      tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
      warpgroup_fence_operand(tCr);
      warpgroup_arrive();
#pragma unroll
      for (int ik = 0; ik < size<2>(tBr); ++ik) {
        cute::gemm(tiled_mma,
                   tBr(_, _, ik, ismem_read),
                   tAr(_, _, ik, ismem_read),
                   tCr);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(tCr);

      // Per-column y-scale = xscale[ik, m_for_this_col] * wscale[this_k_block].
      //
      // P0 register reduction (2026-05-12): previously we precomputed kN
      // floats in tCS[] then FMA'd in a separate loop. For kTileM=48/64
      // that's 24/32 fp32 of extra live state on top of tCr+tDr → ptxas
      // spilled. Fold scale-read directly into the FMA loop:
      //   • wscale_val: 1 fp32 reused across kN
      //   • xs_mine:    1 fp32 broadcast via __shfl (kTileM≤32) or
      //                 sAS reads inside the loop (kTileM>32)
      // Inner shfl/load latency is ~4 cycles, fully hidden by FMA throughput.
      // Skip entirely if kAblNoDequant (timing-only path).
      auto tDr_mn = retile_fragment(tDr);
      if constexpr (!kAblNoDequant) {
        const float wscale_val = sBS(itile);  // ik == itile for kTileK=128

        if constexpr (Config::kTileM <= 32) {
          // Warp-shuffle path: each lane holds one sAS row, broadcast via __shfl.
          const int lane = idx & 31;
          const int my_m = lane % Config::kTileM;
          const float xs_mine = sAS(my_m, 0, ismem_read);
#pragma unroll
          for (int in = 0; in < kN; ++in) {
            const int m_idx = get<1>(tI_mn(0, in));
            const float yscale =
                __shfl_sync(0xffffffff, xs_mine, m_idx) * wscale_val;
#pragma unroll
            for (int im = 0; im < kM; ++im) {
              tDr_mn(im, in) = tCr_mn(im, in) * yscale + tDr_mn(im, in);
            }
          }
        } else {
          // Direct shmem read for kTileM > 32 (kTileM=48/64). The smem
          // load is per-(in) so the compiler can interleave it with FMA.
#pragma unroll
          for (int in = 0; in < kN; ++in) {
            const int m_idx = get<1>(tI_mn(0, in));
            const float yscale = sAS(m_idx, 0, ismem_read) * wscale_val;
#pragma unroll
            for (int im = 0; im < kM; ++im) {
              tDr_mn(im, in) = tCr_mn(im, in) * yscale + tDr_mn(im, in);
            }
          }
        }
      } else {
        // No-dequant path: plain accumulate (preserves data dependency on
        // tCr so the wgmma is not DCE-ed). Numerically wrong — for timing only.
#pragma unroll
        for (int in = 0; in < kN; ++in) {
#pragma unroll
          for (int im = 0; im < kM; ++im) {
            tDr_mn(im, in) = tCr_mn(im, in) + tDr_mn(im, in);
          }
        }
      }
    }

    // Drain any in-flight cp.async before reusing smem for the C tile.
    cp_async_wait<0>();
    __syncthreads();

    // ───────── Epilogue ─────────
    auto tCrh = make_tensor_like<cute::bfloat16_t>(tCr);
#pragma unroll
    for (int i = 0; i < size(tCr); ++i) {
      tCrh(i) = static_cast<Tout>(tDr(i));
    }

    using R2SCopyAtomC = typename Config::R2SCopyAtomC;
    auto tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
    auto thr_copy_c = tiled_copy_c.get_slice(idx);
    auto tCr4s = thr_copy_c.retile_S(tCrh);
    auto tCs4r = thr_copy_c.partition_D(sCT);

    cute::copy(tiled_copy_c, tCr4s, tCs4r);
    __syncthreads();
    cute::copy_if(s2g_tiled_copy_c, pred_c, tCs2g, tCg2s);
    __syncthreads();
  }

  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

}  // namespace kernels

void group_gemm_fp8_blockwise_scatter_async(
    void *y_ptr, const void *x_ptr, const void *w_ptr,
    const void *xscale_ptr, const void *wscale_ptr,
    const void *row_indices_ptr,
    const void *seqlens_ptr, const void *cu_seqlens_ptr,
    const void *tiles_ptr, const void *cu_tiles_ptr,
    const void *task_map_ptr, int task_map_len,
    int m, int n, int k, int num_group,
    int m_pad, int num_block_n, int num_block_k_pad4,
    int num_seq_per_group_avg, bool use_pdl, cudaStream_t stream) {
  using namespace cute;  // NOLINT

  using Tin = cute::float_e4m3_t;
  using Tout = cute::bfloat16_t;

  // Tile/scale alignment contracts (match the non-scatter blockwise variant).
  assert(n % 64 == 0 &&
         "blockwise scatter cp.async: n must be a multiple of 64");
  assert(k % 128 == 0 &&
         "blockwise scatter cp.async: k must be a multiple of 128");
  assert((k / 128) <= 128 &&
         "blockwise scatter cp.async: k/128 (ntile_k) must be <= 128");

  // Every per-(m, n, k) winner from tune_bws_per_shape.py v2 (27 / 27
  // shapes) chose kTileN=64 and kTileK=128, so we hard-wire those here
  // and only vary (kStage, kMinBlocks, kGridMul) per shape.
  constexpr int kTileN = 64;
  constexpr int kTileK = 128;

  const int num_tile_n = (n + kTileN - 1) / kTileN;
  cutlass::FastDivmod flat_divider(num_tile_n);

  // launch: instantiates the kernel for a fixed (kTileM, kStage,
  // kMinBlocks). gridMul is a runtime int because it only affects
  // grid size, not the kernel template — so we don't have to compile
  // a separate kernel for every gridMul value the picker emits.
  auto launch = [&](auto tile_m_tag, auto stage_tag, auto kminblk_tag,
                    int gridMul) {
    constexpr int kTileM = decltype(tile_m_tag)::value;
    constexpr int kStage = decltype(stage_tag)::value;
    constexpr int kMinBlocks = decltype(kminblk_tag)::value;

    using GemmConfig =
        config::BlockWiseFP8GemmConfig<Tin, Tout, kTileM, kTileN, kTileK,
                                       kStage>;
    GemmConfig gemm_config;

    dim3 block(128);
    // Schedule scratch: shm_tiles (num_group+1 ints) + shm_row_indices
    // (kTileM ints) appended after the compute SMEM block.
    const int shm_size = gemm_config.kShmSize +
                         (num_group + 1) * sizeof(int) +
                         kTileM * sizeof(int);
    dim3 grid(get_sm_count() * gridMul);
    const bool use_task_map = (task_map_ptr != nullptr);

    auto dispatch = [&](auto use_task_map_tag, auto use_pdl_tag) {
      constexpr bool kUseTaskMap = decltype(use_task_map_tag)::value;
      constexpr bool kUsePDL = decltype(use_pdl_tag)::value;

      // ABLATION: read once per launch (cheap, cached after first call).
      static int abl_mask_cached = []() {
        const char *e = std::getenv("HPC_BWS_ABLATE");
        return e ? atoi(e) : 0;
      }();
      const int abl_mask_runtime = abl_mask_cached;

      auto launch_with_mask = [&](auto mask_tag) {
        constexpr int kAblMask = decltype(mask_tag)::value;
        auto kernel =
            kernels::group_gemm_fp8_blockwise_scatter_kernel<GemmConfig,
                                                             kUseTaskMap,
                                                             kUsePDL,
                                                             kMinBlocks,
                                                             kAblMask>;
        cudaFuncSetAttribute(kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             shm_size);
        const int4 *tm_ptr_typed =
            kUseTaskMap ? reinterpret_cast<const int4 *>(task_map_ptr)
                        : nullptr;
        int tm_ub = kUseTaskMap ? task_map_len : 0;
        if constexpr (kUsePDL) {
          cudaLaunchAttribute attr[1];
          attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
          attr[0].val.programmaticStreamSerializationAllowed = 1;
          cudaLaunchConfig_t cfg{};
          cfg.gridDim = grid;
          cfg.blockDim = block;
          cfg.dynamicSmemBytes = shm_size;
          cfg.stream = stream;
          cfg.attrs = attr;
          cfg.numAttrs = 1;
          cudaLaunchKernelEx(&cfg, kernel, y_ptr, x_ptr, w_ptr,
                             reinterpret_cast<const float *>(xscale_ptr),
                             reinterpret_cast<const float *>(wscale_ptr),
                             reinterpret_cast<const int *>(row_indices_ptr),
                             reinterpret_cast<const int *>(seqlens_ptr),
                             reinterpret_cast<const int *>(cu_seqlens_ptr),
                             reinterpret_cast<const int *>(tiles_ptr),
                             reinterpret_cast<const int *>(cu_tiles_ptr),
                             tm_ptr_typed, tm_ub, m, n, k, num_group,
                             m_pad, num_block_n, num_block_k_pad4,
                             flat_divider);
        } else {
          kernel<<<grid, block, shm_size, stream>>>(
              y_ptr, x_ptr, w_ptr,
              reinterpret_cast<const float *>(xscale_ptr),
              reinterpret_cast<const float *>(wscale_ptr),
              reinterpret_cast<const int *>(row_indices_ptr),
              reinterpret_cast<const int *>(seqlens_ptr),
              reinterpret_cast<const int *>(cu_seqlens_ptr),
              reinterpret_cast<const int *>(tiles_ptr),
              reinterpret_cast<const int *>(cu_tiles_ptr),
              tm_ptr_typed, tm_ub, m, n, k, num_group,
              m_pad, num_block_n, num_block_k_pad4, flat_divider);
        }
      };

      // 8 ablation buckets: mask in [0..7]. Default (0) is the only one
      // compiled in production; the others are gated by a compile-time
      // macro to avoid bloating non-ablation builds.
      //
      // DEBUG TRIM 2026-05-12: temporarily disable the ablation expansion
      // so only mask=0 is instantiated. This cuts compile time of this TU
      // by ~8× while we iterate on register-pressure / pipeline fixes.
      // To re-enable ablation, define HPC_BWS_ABLATION_EXPAND.
#ifdef HPC_BWS_ABLATION_EXPAND
      switch (abl_mask_runtime) {
        case 1: launch_with_mask(std::integral_constant<int, 1>{}); break;
        case 2: launch_with_mask(std::integral_constant<int, 2>{}); break;
        case 3: launch_with_mask(std::integral_constant<int, 3>{}); break;
        case 4: launch_with_mask(std::integral_constant<int, 4>{}); break;
        case 5: launch_with_mask(std::integral_constant<int, 5>{}); break;
        case 6: launch_with_mask(std::integral_constant<int, 6>{}); break;
        case 7: launch_with_mask(std::integral_constant<int, 7>{}); break;
        default: launch_with_mask(std::integral_constant<int, 0>{}); break;
      }
#else
      (void)abl_mask_runtime;
      launch_with_mask(std::integral_constant<int, 0>{});
#endif
    };

    if (use_task_map) {
      if (use_pdl) dispatch(std::true_type{}, std::true_type{});
      else         dispatch(std::true_type{}, std::false_type{});
    } else {
      if (use_pdl) dispatch(std::false_type{}, std::true_type{});
      else         dispatch(std::false_type{}, std::false_type{});
    }
  };

  // ─── Per-(kTileM, m_avg, n, k) winners from tune_bws_per_shape v3 ───
  //
  // Tuned 2026-05-11 over the test_fuse_moe_cp_async grid (g=192,
  // k=4096, n ∈ {384, 512, 1024}) plus 3 production fuse_moe shapes
  // (g=192/48/24 m_avg=5/21/42 n=384), with EXPANDED ranges:
  //   kMinBlocks ∈ {2..8}  (was {2..5})
  //   kStage     ∈ {2..4}  (unchanged)
  // For shapes outside the table we fall back to a per-tile_m default.
  //
  // Every tuned winner uses kTileN=64. Picker emits (kStage, kMinBlocks,
  // gridMul) per shape.
  auto pick_and_launch = [&](auto tile_m_tag) {
    constexpr int kTileM = decltype(tile_m_tag)::value;

    // Per-tile_m fallback (used when (m_avg, n, k) is not in the table).
    constexpr int fbKs  = 2;
    constexpr int fbKmb = (kTileM == 8)  ? 3
                        : (kTileM == 16) ? 2
                        : (kTileM == 32) ? 5
                        : (kTileM == 48) ? 4
                        :                  4;
    constexpr int fbGm  = (kTileM == 8)  ? 10
                        : (kTileM == 16) ? 5
                        : (kTileM == 32) ? 14
                        : (kTileM == 48) ? 8
                        :                 12;

    int defKs  = fbKs;
    int defKmb = fbKmb;
    int defGm  = fbGm;

    if (k == 4096) {
      const int m_avg = num_seq_per_group_avg;
      if constexpr (kTileM == 8) {
        if (m_avg <= 1) {
          if      (n ==  384) { defKs = 2; defKmb = 3; defGm = 10; }
          else if (n ==  512) { defKs = 4; defKmb = 2; defGm = 13; }
          else if (n ==  768) { defKs = 4; defKmb = 2; defGm = 12; }   // Hunyuan TP=4 GEMM1 BS≤32
          else if (n == 1024) { defKs = 4; defKmb = 2; defGm = 10; }
        } else if (m_avg <= 2) {
          if      (n ==  384) { defKs = 2; defKmb = 2; defGm = 12; }
          else if (n ==  512) { defKs = 4; defKmb = 2; defGm = 15; }
          else if (n ==  768) { defKs = 4; defKmb = 2; defGm = 18; }   // Hunyuan TP=4 GEMM1 BS=64
          else if (n == 1024) { defKs = 4; defKmb = 2; defGm = 10; }
        } else if (m_avg <= 5) {
          if      (n ==  384) { defKs = 2; defKmb = 2; defGm = 10; }
          else if (n ==  512) { defKs = 4; defKmb = 2; defGm = 14; }
          else if (n ==  768) { defKs = 4; defKmb = 2; defGm = 18; }   // Hunyuan TP=4 GEMM1 BS=128
          else if (n == 1024) { defKs = 4; defKmb = 2; defGm = 10; }
        } else { // m_avg in (5, 8]
          if      (n ==  384) { defKs = 2; defKmb = 3; defGm = 11; }
          else if (n ==  512) { defKs = 4; defKmb = 2; defGm = 14; }
          else if (n ==  768) { defKs = 4; defKmb = 2; defGm = 18; }   // Hunyuan TP=4 GEMM1 (extrapolated)
          else if (n == 1024) { defKs = 4; defKmb = 2; defGm = 10; }
        }
      } else if constexpr (kTileM == 16) {
        if (m_avg <= 10) {
          if      (n ==  384) { defKs = 2; defKmb = 2; defGm =  5; }
          else if (n ==  512) { defKs = 2; defKmb = 2; defGm = 14; }
          else if (n ==  768) { defKs = 2; defKmb = 2; defGm = 18; }   // Hunyuan TP=4 GEMM1 BS=256
          else if (n == 1024) { defKs = 2; defKmb = 4; defGm = 14; }
        } else { // m_avg in (10, 16]
          if      (n ==  384) { defKs = 2; defKmb = 2; defGm =  5; }
          else if (n ==  512) { defKs = 2; defKmb = 2; defGm = 13; }
          else if (n ==  768) { defKs = 2; defKmb = 2; defGm = 18; }   // Hunyuan TP=4 GEMM1 (extrapolated)
          else if (n == 1024) { defKs = 2; defKmb = 4; defGm = 14; }
        }
      } else if constexpr (kTileM == 32) {
        if (m_avg <= 21) {
          if      (n ==  384) { defKs = 4; defKmb = 2; defGm =  6; }
          else if (n ==  512) { defKs = 2; defKmb = 2; defGm = 15; }
          else if (n ==  768) { defKs = 2; defKmb = 2; defGm = 10; }   // Hunyuan TP=4 GEMM1 BS=512
          else if (n == 1024) { defKs = 2; defKmb = 2; defGm = 14; }
        } else { // m_avg in (21, 32]
          if      (n ==  384) { defKs = 2; defKmb = 7; defGm =  7; }
          else if (n ==  512) { defKs = 2; defKmb = 5; defGm = 14; }
          else if (n ==  768) { defKs = 2; defKmb = 2; defGm = 10; }   // Hunyuan TP=4 GEMM1 (extrapolated)
          else if (n == 1024) { defKs = 2; defKmb = 5; defGm = 14; }
        }
      } else if constexpr (kTileM == 48) {
        if      (n ==  384) { defKs = 3; defKmb = 4; defGm =  2; }
        else if (n ==  512) { defKs = 3; defKmb = 4; defGm =  4; }
        else if (n ==  768) { defKs = 3; defKmb = 4; defGm = 12; }   // Hunyuan TP=4 GEMM1 BS=1024 m_avg=42
        else if (n == 1024) { defKs = 3; defKmb = 4; defGm =  8; }
      } else if constexpr (kTileM == 64) {
        if      (n ==  384) { defKs = 2; defKmb = 4; defGm = 12; }
        else if (n ==  512) { defKs = 2; defKmb = 2; defGm =  4; }
        else if (n ==  768) { defKs = 2; defKmb = 4; defGm = 12; }   // Hunyuan TP=4 GEMM1 BS=2048 m_avg=85
        else if (n == 1024) { defKs = 2; defKmb = 2; defGm =  8; }
        else if (n == 14336) { defKs = 2; defKmb = 3; defGm =  8; }   // Mixtral GEMM1 m_avg>48
      }
      // Mixtral GEMM1 (n=14336): not in standard {384,512,1024} table.
      // Tuned 2026-05-11 — winners at m_avg ≤ 16. m_avg ≥ 32 left to fallback.
      if (n == 14336) {
        if constexpr (kTileM == 8) {
          if      (m_avg <= 1) { defKs = 2; defKmb = 4; defGm = 16; }
          else if (m_avg <= 8) { defKs = 2; defKmb = 3; defGm = 16; }
        } else if constexpr (kTileM == 16) {
          defKs = 4; defKmb = 4; defGm = 14;
        }
      }
    }

    // DeepSeek-V3 TP=8 GEMM1: k=7168 long K with n=512.
    // Tuned 2026-05-11 — winners at m_avg ≤ 16. m_avg ≥ 32 left to fallback.
    // Note: fuse_moe calls this with m_avg=0 (topk routing, sparse experts);
    // (Ks=2 Kmb=2 Gm=8) wins on that pattern AND on standalone m=1 (within
    // 1% of the dense-m=1 winner Ks=4 Kmb=4 Gm=10), so we use Ks=2 here.
    if (k == 7168 && n == 512) {
      const int m_avg = num_seq_per_group_avg;
      if constexpr (kTileM == 8) {
        // BS-→m_avg(int) mapping for DeepSeek (E=160, topk=6):
        //   BS=16  → m_avg=0 (96/160)
        //   BS=32  → m_avg=1 (192/160)
        //   BS=64  → m_avg=2 (384/160)
        //   BS=128 → m_avg=4 (768/160)
        // Tuned 2026-05-12 in fresh-subprocess sweep, 400 reps each.
        if      (m_avg <= 0) { defKs = 2; defKmb = 2; defGm =  8; }  // BS=16: tied with everything
        else if (m_avg <= 1) { defKs = 3; defKmb = 2; defGm =  8; }  // BS=32: −5.6 μs
        else if (m_avg <= 3) { defKs = 2; defKmb = 3; defGm = 10; }  // BS=64: −3.5 μs
        else if (m_avg <= 8) { defKs = 4; defKmb = 2; defGm = 10; }  // BS=128: already optimal
      } else if constexpr (kTileM == 16) {
        defKs = 2; defKmb = 2; defGm = 14;
      } else if constexpr (kTileM == 48) {
        // m_avg in (32, 48]: tuned 2026-05-11 for DeepSeek-V3 BS=1024
        // (m_avg=38, ng=160). Ks=2 Kmb=2 Gm=8 wins at m=38 (294.66 μs).
        // Combined with bw kTileM=48 winner (Ks=2 Kmb=5 Gm=5) this
        // brings BS=1024 fuse_moe from 696 μs (picker miss) → 540 μs
        // (cpa beats TMA 0.98×). bws contribution to that win is small
        // (~−3 μs over fallback bws); bw is the dominant fix.
        defKs = 2; defKmb = 2; defGm = 8;
      }
    }

    // ─── Tune-mode env-var override ─────────────────────────────────
    // For autotune sweeps only: HPC_BWS_KSTAGE_<TM> / HPC_BWS_KMINBLK_<TM> /
    // HPC_BWS_GRIDMUL_<TM> override (defKs, defKmb, defGm). Production
    // builds never set these. Reading getenv per call is ~50ns vs ~80μs
    // kernel.
    {
      char nm[40];
      const int tm = kTileM;
      std::snprintf(nm, sizeof(nm), "HPC_BWS_KSTAGE_%d", tm);
      if (const char *s = std::getenv(nm)) defKs  = std::atoi(s);
      std::snprintf(nm, sizeof(nm), "HPC_BWS_KMINBLK_%d", tm);
      if (const char *s = std::getenv(nm)) defKmb = std::atoi(s);
      std::snprintf(nm, sizeof(nm), "HPC_BWS_GRIDMUL_%d", tm);
      if (const char *s = std::getenv(nm)) defGm  = std::atoi(s);
    }

    // Compile-time-instantiate (kStage, kMinBlocks) pairs.
    //
    // DEBUG TRIM 2026-05-12: aggressively pruned to ONLY the combos hit by
    // the Hunyuan TP=4 BS=32 production path so we can iterate on register-
    // pressure fixes quickly. Other shapes assert at launch.
    //   • Hunyuan BS=32: kTileM=8, k=4096, n∈{768,4096}: both fall through
    //     to fbKs=2/fbKmb=3/fbGm=10  → (kStage=2, kMinBlocks=3)
    // To restore the full sweep table for autotune, define HPC_BWS_FULL_TABLE.
#define HPC_BWS_DISPATCH(KS, KMB)                                        \
    if (defKs == (KS) && defKmb == (KMB)) {                              \
      launch(Int<kTileM>{}, Int<(KS)>{}, Int<(KMB)>{}, defGm);           \
      return;                                                            \
    }

#ifdef HPC_BWS_FULL_TABLE
    if constexpr (kTileM == 8) {
      HPC_BWS_DISPATCH(2, 2) HPC_BWS_DISPATCH(2, 3) HPC_BWS_DISPATCH(2, 4)
      HPC_BWS_DISPATCH(2, 5) HPC_BWS_DISPATCH(2, 6) HPC_BWS_DISPATCH(2, 7)
      HPC_BWS_DISPATCH(2, 8)
      HPC_BWS_DISPATCH(3, 2) HPC_BWS_DISPATCH(3, 3) HPC_BWS_DISPATCH(3, 4)
      HPC_BWS_DISPATCH(3, 5) HPC_BWS_DISPATCH(3, 6)
      HPC_BWS_DISPATCH(4, 2) HPC_BWS_DISPATCH(4, 3) HPC_BWS_DISPATCH(4, 4)
      HPC_BWS_DISPATCH(4, 5)
    } else if constexpr (kTileM == 16) {
      HPC_BWS_DISPATCH(2, 2) HPC_BWS_DISPATCH(2, 3) HPC_BWS_DISPATCH(2, 4)
      HPC_BWS_DISPATCH(2, 5) HPC_BWS_DISPATCH(2, 6) HPC_BWS_DISPATCH(2, 7)
      HPC_BWS_DISPATCH(2, 8)
      HPC_BWS_DISPATCH(3, 2) HPC_BWS_DISPATCH(3, 3) HPC_BWS_DISPATCH(3, 4)
      HPC_BWS_DISPATCH(3, 5)
      HPC_BWS_DISPATCH(4, 2) HPC_BWS_DISPATCH(4, 3) HPC_BWS_DISPATCH(4, 4)
    } else if constexpr (kTileM == 32) {
      HPC_BWS_DISPATCH(2, 2) HPC_BWS_DISPATCH(2, 3) HPC_BWS_DISPATCH(2, 4)
      HPC_BWS_DISPATCH(2, 5) HPC_BWS_DISPATCH(2, 6) HPC_BWS_DISPATCH(2, 7)
      HPC_BWS_DISPATCH(3, 2) HPC_BWS_DISPATCH(3, 3) HPC_BWS_DISPATCH(3, 4)
      HPC_BWS_DISPATCH(3, 5)
      HPC_BWS_DISPATCH(4, 2) HPC_BWS_DISPATCH(4, 3)
    } else if constexpr (kTileM == 48) {
      // Larger accumulator — kmb≥7 spills; cap at 6.
      HPC_BWS_DISPATCH(2, 2) HPC_BWS_DISPATCH(2, 3) HPC_BWS_DISPATCH(2, 4)
      HPC_BWS_DISPATCH(2, 5) HPC_BWS_DISPATCH(2, 6)
      HPC_BWS_DISPATCH(3, 2) HPC_BWS_DISPATCH(3, 3) HPC_BWS_DISPATCH(3, 4)
    } else if constexpr (kTileM == 64) {
      // Largest accumulator — kmb≥5 spills; cap at 4.
      HPC_BWS_DISPATCH(2, 2) HPC_BWS_DISPATCH(2, 3) HPC_BWS_DISPATCH(2, 4)
      HPC_BWS_DISPATCH(3, 2) HPC_BWS_DISPATCH(3, 3) HPC_BWS_DISPATCH(3, 4)
    }
#else  // !HPC_BWS_FULL_TABLE — debug trim
    if constexpr (kTileM == 8) {
      // Cover the Hunyuan BS=32 fallback (2,3) plus a couple of common
      // sweep combos so we can override via env vars without recompiling.
      HPC_BWS_DISPATCH(2, 2) HPC_BWS_DISPATCH(2, 3) HPC_BWS_DISPATCH(2, 4)
      HPC_BWS_DISPATCH(3, 2) HPC_BWS_DISPATCH(3, 3)
      HPC_BWS_DISPATCH(4, 2)
    } else if constexpr (kTileM == 16) {
      HPC_BWS_DISPATCH(2, 2) HPC_BWS_DISPATCH(2, 3)
    } else if constexpr (kTileM == 32) {
      HPC_BWS_DISPATCH(2, 2) HPC_BWS_DISPATCH(2, 5)
    } else if constexpr (kTileM == 48) {
      HPC_BWS_DISPATCH(2, 4) HPC_BWS_DISPATCH(3, 4)
    } else if constexpr (kTileM == 64) {
      HPC_BWS_DISPATCH(2, 4)
    }
#endif  // HPC_BWS_FULL_TABLE
#undef HPC_BWS_DISPATCH

    assert(false &&
           "blockwise scatter cp.async: picker produced an "
           "(kStage, kMinBlocks) combo that is not compiled in.");
  };

  if (num_seq_per_group_avg <= 8) {
    pick_and_launch(Int<8>{});
  } else if (num_seq_per_group_avg <= 16) {
    pick_and_launch(Int<16>{});
  } else if (num_seq_per_group_avg <= 32) {
    pick_and_launch(Int<32>{});
  } else if (num_seq_per_group_avg <= 48) {
    pick_and_launch(Int<48>{});
  } else {
    pick_and_launch(Int<64>{});
  }
}

}  // namespace group_gemm_cp_async
}  // namespace hpc
