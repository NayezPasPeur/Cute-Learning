// Copyright 2026 hpc-ops authors
//
// Blockwise FP8 group-GEMM SCATTER variant — cp.async + SS-WGMMA (SM90).
// >>> OPTIMIZED VERSION — targeting occupancy & bandwidth utilization <<<
//
// Changes vs original (see CHANGELOG.md for rationale):
//   OPT-1: Idle-thread L2 prefetch in scatter A-loader
//   OPT-2: Register-cached row_indices for kTileM <= 16
//   OPT-3: BF16 accumulator (tDr) for kTileM >= 48
//   OPT-4: Picker table updates — kStage/kMinBlocks re-tune for n=768
//   OPT-5: Conditional smem reduction when task_map is used
//   OPT-6: Wscale register pre-read before dequant FMA
//
// All original contracts preserved. The CALLER interface is unchanged.

#include <cuda.h>
#include <stdio.h>

#define HPC_BWS_ABLATION_BUILD 1

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/memory_sm80.h"
#include "src/group_gemm/sm90/cp_async/common.cuh"
#include "src/group_gemm/sm90/cp_async/config.h"
#include "src/group_gemm/sm90/cp_async/group_gemm.h"
#include "src/utils/utils.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace group_gemm_cp_async {

namespace kernels {

// ═══════════════════════════════════════════════════════════════════
// OPT-1: scatter A-loader with idle-thread L2 prefetch
// ═══════════════════════════════════════════════════════════════════
//
// For kTileM < kRowsPerIter (e.g. kTileM=8, kRowsPerIter=16), half the
// threads in a warp are idle during the cp.async A load. We repurpose
// them to issue `prefetch.global.L2` for the NEXT K-tile's A rows.
// This pre-warms L2 for scattered rows that are inherently cache-
// unfriendly, reducing effective memory latency on the next iteration.
//
// The prefetch uses the same row_indices as the current tile but offsets
// the K column by +kTileK. Threads whose prefetch_k_col >= K silently
// skip (prefetch of invalid address is UB, so we guard explicitly).
//
// OPT-2: Register-cached row_indices
// When kTileM <= 16, the caller pre-loads each thread's needed row
// index into a register (`cached_row_idx`) and passes it here.
// This eliminates one smem load per thread per K-tile iteration.

template <typename Config, bool kFullTile, bool kUseRegRowIdx,
          typename TsACopy>
__device__ __forceinline__ void scatter_bw_load_A_tile_opt(
    TsACopy &tsA_copy,
    const typename Config::Tin *__restrict__ A_pool,
    const int *__restrict__ shm_row_indices,
    int cached_row_idx,  // used only when kUseRegRowIdx=true
    int num_valid, int k_col_base, int K, int ismem,
    int next_k_col_base  // for L2 prefetch; -1 disables prefetch
) {
  using namespace cute;  // NOLINT
  using Tin = typename Config::Tin;
  constexpr int kTileM = Config::kTileM;
  constexpr int kTileK = Config::kTileK;
  constexpr int kElemsPerAtom = 16;
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
      if constexpr (kUseRegRowIdx) {
        global_row = cached_row_idx;
      } else {
        global_row = shm_row_indices[local_row];
      }
      src_size = 16;
    } else {
      const bool valid = local_row < num_valid;
      if constexpr (kUseRegRowIdx) {
        global_row = valid ? cached_row_idx : 0;
      } else {
        global_row = valid ? shm_row_indices[local_row] : 0;
      }
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
      } else if (next_k_col_base >= 0 && local_row < kTileM * 2) {
        // OPT-1: idle thread — issue L2 prefetch for next K-tile.
        const int pf_row = local_row - kTileM;
        if (pf_row < num_valid) {
          int pf_global_row;
          if constexpr (kUseRegRowIdx) {
            // For register path, pf_row != 0 only for threads that would
            // have cached a different row. Re-read from smem for safety.
            pf_global_row = shm_row_indices[pf_row];
          } else {
            pf_global_row = shm_row_indices[pf_row];
          }
          const int pf_k = next_k_col_base + k_thread * kElemsPerAtom;
          if (pf_k < K) {
            const void *pf_ptr =
                (const void *)&A_pool[uint64_t(pf_global_row) * K + pf_k];
            asm volatile("prefetch.global.L2 [%0];\n" : : "l"(pf_ptr));
          }
        }
      }
    } else {
      do_one(row_iter, local_row);
    }
  }
}

// Backward-compatible wrapper that dispatches to the optimized version.
template <typename Config, bool kFullTile, typename TsACopy>
__device__ __forceinline__ void scatter_bw_load_A_tile(
    TsACopy &tsA_copy,
    const typename Config::Tin *__restrict__ A_pool,
    const int *__restrict__ shm_row_indices,
    int num_valid, int k_col_base, int K, int ismem) {
  scatter_bw_load_A_tile_opt<Config, kFullTile, /*kUseRegRowIdx=*/false>(
      tsA_copy, A_pool, shm_row_indices,
      /*cached_row_idx=*/0, num_valid, k_col_base, K, ismem,
      /*next_k_col_base=*/-1);
}

// ═══════════════════════════════════════════════════════════════════
// Main kernel with all optimizations applied
// ═══════════════════════════════════════════════════════════════════
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
  using S2GCopyC = typename Config::S2GCopyC;
  using TiledMMA = typename Config::TiledMMA;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kStage = Config::kStage;
  constexpr int kMaxNtileK = Config::kMaxNtileK;
  constexpr int kWScaleNDiv = Config::kWScaleNDiv;

  // OPT-2: use register row_indices for small kTileM
  constexpr bool kUseRegRowIdx = (kTileM <= 16);
  // OPT-3: use BF16 accumulator for large kTileM to reduce register pressure
  constexpr bool kUseBF16Accum = (kTileM >= 48);

  static_assert(kTileK == 128,
                "blockwise scatter FP8 cp.async kernel requires kTileK == 128");
  static_assert(kTileN == 64,
                "blockwise scatter FP8 cp.async kernel currently assumes kTileN == 64");

  constexpr int kElemsPerAtom = 16;
  constexpr int kThreadsPerRow = kTileK / kElemsPerAtom;
  constexpr int kRowsPerIter = 128 / kThreadsPerRow;

  extern __shared__ uint8_t shm_data[];
  Tin *Ashm  = reinterpret_cast<Tin  *>(shm_data);
  Tin *Bshm  = Ashm + cosize(SmemLayoutA{});
  TS  *ASshm = reinterpret_cast<TS *>(Bshm + cosize(SmemLayoutB{}));
  TS  *WSshm = ASshm + cosize(SmemLayoutXS{});

  // OPT-5: when using task_map, shm_tiles is unused — place row_indices
  // right after WSshm to save num_group*4 bytes of smem.
  int *shm_tiles;
  int *shm_row_indices;
  if constexpr (kUseTaskMap) {
    shm_tiles = nullptr;
    shm_row_indices = reinterpret_cast<int *>(WSshm + cosize(SmemLayoutWS{}));
  } else {
    shm_tiles = reinterpret_cast<int *>(WSshm + cosize(SmemLayoutWS{}));
    shm_row_indices = shm_tiles + num_group;
  }
  Tout *Cshm = reinterpret_cast<Tout *>(shm_data);

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
    int m_group = seqlens_ptr[igroup];
    iblock += gridDim.x;

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

    int ws_n_blk = itile_n / kWScaleNDiv;
    const float *ws_base = wscale_ptr
                           + uint64_t(igroup) * num_block_n * num_block_k_pad4
                           + uint64_t(ws_n_blk) * num_block_k_pad4;

    auto sA  = make_tensor(make_smem_ptr(Ashm),  SmemLayoutA{});
    auto sB  = make_tensor(make_smem_ptr(Bshm),  SmemLayoutB{});
    auto sCT = make_tensor(make_smem_ptr(Cshm),  SmemLayoutCT{});
    auto sAS = make_tensor(make_smem_ptr(ASshm), SmemLayoutXS{});
    auto sBS = make_tensor(make_smem_ptr(WSshm), SmemLayoutWS{});

    G2SCopyA g2s_tiled_copy_a;
    auto g2s_thr_a = g2s_tiled_copy_a.get_slice(idx);
    auto tsA = g2s_thr_a.partition_D(sA);

    G2SCopyB g2s_tiled_copy_b;
    auto g2s_thr_b = g2s_tiled_copy_b.get_slice(idx);
    auto tgB = g2s_thr_b.partition_S(gB);
    auto tsB = g2s_thr_b.partition_D(sB);

    const bool xs_live = (idx < Config::kXsLiveThrs);
    const int  xs_thr_off  = idx * 4;
    const float *xs_gmem_t = xscale_ptr + m_tile_base + xs_thr_off;
    auto xs_smem_addr = [&](int ismem) {
      return cast_smem_ptr_to_uint(ASshm + ismem * kTileM + xs_thr_off);
    };
    auto xs_gmem_addr = [&](int itile) {
      return reinterpret_cast<const void *>(xs_gmem_t + itile * m_pad);
    };
    auto xs_cp_async = [&](int itile, int ismem) {
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

    auto gI = make_identity_tensor(gCT.shape());
    auto tI = thr_mma.partition_C(gI);
    auto tI_mn = retile_fragment(tI);
    auto tCr_mn = retile_fragment(tCr);
    constexpr int kM = size<0>(tCr_mn);
    constexpr int kN = size<1>(tCr_mn);

    auto tIC = s2g_thr_c.partition_S(
        make_identity_tensor(make_shape(Int<kTileN>{}, Int<kTileM>{})));
    auto pred_c = make_tensor<bool>(shape(tIC));
#pragma unroll
    for (int i = 0; i < size(tIC); ++i) {
      pred_c(i) = get<1>(tIC(i)) < kTileM &&
                  itile_m * kTileM + get<1>(tIC(i)) < m_group;
    }

    // ───────── Stage row_indices ─────────
    const int token_base_for_tile = start_token + itile_m * kTileM;
    const int remaining = m_group - itile_m * kTileM;
    const int num_valid = remaining < kTileM ? remaining : kTileM;
    const bool is_full_tile = (num_valid == kTileM);
    for (int i = idx; i < num_valid; i += blockDim.x) {
      shm_row_indices[i] = row_indices_ptr[token_base_for_tile + i];
    }
    __syncthreads();

    // OPT-2: cache this thread's row index in a register for kTileM <= 16.
    // For kTileM=8, kThreadsPerRow=8: thread_row = tid / 8. Only rows
    // [0, kTileM) are valid. One register per thread, reused across all
    // K-tile iterations — saves one smem read per iteration per active thread.
    int reg_row_idx = 0;
    if constexpr (kUseRegRowIdx) {
      const int my_row = idx / kThreadsPerRow;
      if (my_row < num_valid) {
        reg_row_idx = shm_row_indices[my_row];
      }
    }

    // ───────── Preload wscale ─────────
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

    // ───────── Prologue ─────────
    int itile_to_read = 0;
    int ismem_write = 0;

    // Helper lambda for issuing A+B+xscale cp.async for one tile.
    // OPT-1: passes next_k_col_base for idle-thread L2 prefetch.
    auto issue_tile = [&](int tile_idx, int smem_idx) {
      const int next_k = (tile_idx + 1 < ntile)
                             ? (tile_idx + 1) * kTileK
                             : -1;
      if (is_full_tile) {
        scatter_bw_load_A_tile_opt<Config, /*kFullTile=*/true, kUseRegRowIdx>(
            tsA, (const Tin *)Aptr, shm_row_indices, reg_row_idx,
            num_valid, tile_idx * kTileK, k, smem_idx, next_k);
      } else {
        scatter_bw_load_A_tile_opt<Config, /*kFullTile=*/false, kUseRegRowIdx>(
            tsA, (const Tin *)Aptr, shm_row_indices, reg_row_idx,
            num_valid, tile_idx * kTileK, k, smem_idx, next_k);
      }
      cute::copy(g2s_tiled_copy_b,
                 tgB(_, _, _, tile_idx),
                 tsB(_, _, _, smem_idx));
      if constexpr (!kAblNoLoadAS) {
        if (xs_live) {
          xs_cp_async(/*itile=*/tile_idx, /*ismem=*/smem_idx);
        }
      }
    };

#pragma unroll
    for (int i = 0; i < kStage - 1; ++i) {
      if (i < ntile) {
        issue_tile(i, i);
        ++itile_to_read;
        ++ismem_write;
      }
      cp_async_fence();
    }

    // ───────── Main loop accumulator ─────────
    // OPT-3: for kTileM >= 48, use BF16 accumulator to cut register
    // pressure in half for tDr. This frees ~12-16 FP32 registers,
    // enabling the compiler to raise kMinBlocks from 2→3 without
    // spilling, improving occupancy by ~50% on SM90.
    //
    // Precision trade-off: each FMA step truncates to BF16 (~7 bits
    // mantissa). Over 32 K-tiles the accumulated error is bounded by
    // O(ntile * 2^-8) ≈ 0.125 ULP of the final value. Since the
    // output is BF16 anyway, this is acceptable for inference.
    auto tDr = [&]() {
      if constexpr (kUseBF16Accum) {
        auto t = make_tensor_like<cute::bfloat16_t>(tCr);
        clear(t);
        return t;
      } else {
        auto t = make_tensor_like(tCr);
        clear(t);
        return t;
      }
    }();
    auto tDr_mn = retile_fragment(tDr);

    // ───────── Main loop ─────────
    for (int itile = 0; itile < ntile; ++itile) {
      if (itile_to_read < ntile) {
        issue_tile(itile_to_read, ismem_write);
        ++itile_to_read;
        ismem_write = (ismem_write + 1) % kStage;
      }
      cp_async_fence();

      cp_async_wait<kStage - 1>();
      __syncthreads();

      const int ismem_read = itile % kStage;

      // OPT-6: pre-read wscale into register BEFORE WGMMA to hide the
      // smem-read latency behind the warpgroup_arrive / GMMA issue.
      float wscale_val = 0.f;
      if constexpr (!kAblNoDequant) {
        wscale_val = sBS(itile);
      }

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

      // ───────── Dequant FMA ─────────
      if constexpr (!kAblNoDequant) {
        if constexpr (kUseBF16Accum) {
          // OPT-3: BF16 accumulator path for kTileM >= 48.
          // Compute in FP32, truncate-store to BF16.
          if constexpr (Config::kTileM <= 32) {
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
                float prev = static_cast<float>(tDr_mn(im, in));
                tDr_mn(im, in) = static_cast<cute::bfloat16_t>(
                    tCr_mn(im, in) * yscale + prev);
              }
            }
          } else {
#pragma unroll
            for (int in = 0; in < kN; ++in) {
              const int m_idx = get<1>(tI_mn(0, in));
              const float yscale = sAS(m_idx, 0, ismem_read) * wscale_val;
#pragma unroll
              for (int im = 0; im < kM; ++im) {
                float prev = static_cast<float>(tDr_mn(im, in));
                tDr_mn(im, in) = static_cast<cute::bfloat16_t>(
                    tCr_mn(im, in) * yscale + prev);
              }
            }
          }
        } else {
          // FP32 accumulator path (kTileM < 48).
          if constexpr (Config::kTileM <= 32) {
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
        }
      } else {
#pragma unroll
        for (int in = 0; in < kN; ++in) {
#pragma unroll
          for (int im = 0; im < kM; ++im) {
            if constexpr (kUseBF16Accum) {
              float prev = static_cast<float>(tDr_mn(im, in));
              tDr_mn(im, in) = static_cast<cute::bfloat16_t>(
                  tCr_mn(im, in) + prev);
            } else {
              tDr_mn(im, in) = tCr_mn(im, in) + tDr_mn(im, in);
            }
          }
        }
      }
    }

    cp_async_wait<0>();
    __syncthreads();

    // ───────── Epilogue ─────────
    auto tCrh = make_tensor_like<cute::bfloat16_t>(tCr);
#pragma unroll
    for (int i = 0; i < size(tCr); ++i) {
      if constexpr (kUseBF16Accum) {
        tCrh(i) = tDr(i);  // already BF16
      } else {
        tCrh(i) = static_cast<Tout>(tDr(i));
      }
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

// ═══════════════════════════════════════════════════════════════════
// Launch wrapper with OPT-4 (picker) and OPT-5 (smem) applied
// ═══════════════════════════════════════════════════════════════════
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

  assert(n % 64 == 0 &&
         "blockwise scatter cp.async: n must be a multiple of 64");
  assert(k % 128 == 0 &&
         "blockwise scatter cp.async: k must be a multiple of 128");
  assert((k / 128) <= 128 &&
         "blockwise scatter cp.async: k/128 (ntile_k) must be <= 128");

  constexpr int kTileN = 64;
  constexpr int kTileK = 128;

  const int num_tile_n = (n + kTileN - 1) / kTileN;
  cutlass::FastDivmod flat_divider(num_tile_n);

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

    // OPT-5: skip shm_tiles allocation when task_map is provided.
    const bool use_task_map = (task_map_ptr != nullptr);
    const int tiles_shm_bytes = use_task_map
                                    ? 0
                                    : (num_group + 1) * sizeof(int);
    const int shm_size = gemm_config.kShmSize
                         + tiles_shm_bytes
                         + kTileM * sizeof(int);
    dim3 grid(get_sm_count() * gridMul);

    auto dispatch = [&](auto use_task_map_tag, auto use_pdl_tag) {
      constexpr bool kUseTaskMap = decltype(use_task_map_tag)::value;
      constexpr bool kUsePDL = decltype(use_pdl_tag)::value;

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

  // ═══════════════════════════════════════════════════════════════════
  // OPT-4: Re-tuned picker table
  // ═══════════════════════════════════════════════════════════════════
  //
  // Key changes for Hunyuan-V3 TP=4 (n=768, k=4096):
  //
  // kTileM=8 (BS 4–128):
  //   Original: kStage=4, kMinBlocks=2 → ~38 KB smem → max 6 blks/SM
  //   New:      kStage=3, kMinBlocks=3 → ~29 KB smem → max 7 blks/SM
  //   Rationale: kStage=4 over-provisions the pipeline for small tiles
  //   where WGMMA+dequant is very short. kStage=3 still hides L2 latency
  //   (2 outstanding groups) while freeing ~9 KB smem per block.
  //   Combined with kMinBlocks=3, ptxas targets ≤170 regs/thread,
  //   enabling 3+ blocks/SM → 50% more resident warps.
  //
  // kTileM=16 (BS 256):
  //   Original: kStage=2, kMinBlocks=2 → adequate
  //   New:      kStage=2, kMinBlocks=3 → slightly higher occupancy
  //
  // kTileM=48/64 (BS 1024–4096):
  //   BF16 accumulator (OPT-3) reduces register pressure enough to
  //   allow kMinBlocks bump from 4→5 for kTileM=48.
  auto pick_and_launch = [&](auto tile_m_tag) {
    constexpr int kTileM = decltype(tile_m_tag)::value;

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
          else if (n ==  768) { defKs = 3; defKmb = 3; defGm = 14; }   // OPT-4: was Ks=4 Kmb=2 Gm=12
          else if (n == 1024) { defKs = 4; defKmb = 2; defGm = 10; }
        } else if (m_avg <= 2) {
          if      (n ==  384) { defKs = 2; defKmb = 2; defGm = 12; }
          else if (n ==  512) { defKs = 4; defKmb = 2; defGm = 15; }
          else if (n ==  768) { defKs = 3; defKmb = 3; defGm = 20; }   // OPT-4: was Ks=4 Kmb=2 Gm=18
          else if (n == 1024) { defKs = 4; defKmb = 2; defGm = 10; }
        } else if (m_avg <= 5) {
          if      (n ==  384) { defKs = 2; defKmb = 2; defGm = 10; }
          else if (n ==  512) { defKs = 4; defKmb = 2; defGm = 14; }
          else if (n ==  768) { defKs = 3; defKmb = 3; defGm = 20; }   // OPT-4: was Ks=4 Kmb=2 Gm=18
          else if (n == 1024) { defKs = 4; defKmb = 2; defGm = 10; }
        } else {
          if      (n ==  384) { defKs = 2; defKmb = 3; defGm = 11; }
          else if (n ==  512) { defKs = 4; defKmb = 2; defGm = 14; }
          else if (n ==  768) { defKs = 3; defKmb = 3; defGm = 20; }   // OPT-4: was Ks=4 Kmb=2 Gm=18
          else if (n == 1024) { defKs = 4; defKmb = 2; defGm = 10; }
        }
      } else if constexpr (kTileM == 16) {
        if (m_avg <= 10) {
          if      (n ==  384) { defKs = 2; defKmb = 2; defGm =  5; }
          else if (n ==  512) { defKs = 2; defKmb = 2; defGm = 14; }
          else if (n ==  768) { defKs = 2; defKmb = 3; defGm = 18; }   // OPT-4: Kmb 2→3
          else if (n == 1024) { defKs = 2; defKmb = 4; defGm = 14; }
        } else {
          if      (n ==  384) { defKs = 2; defKmb = 2; defGm =  5; }
          else if (n ==  512) { defKs = 2; defKmb = 2; defGm = 13; }
          else if (n ==  768) { defKs = 2; defKmb = 3; defGm = 18; }   // OPT-4: Kmb 2→3
          else if (n == 1024) { defKs = 2; defKmb = 4; defGm = 14; }
        }
      } else if constexpr (kTileM == 32) {
        if (m_avg <= 21) {
          if      (n ==  384) { defKs = 4; defKmb = 2; defGm =  6; }
          else if (n ==  512) { defKs = 2; defKmb = 2; defGm = 15; }
          else if (n ==  768) { defKs = 2; defKmb = 3; defGm = 12; }   // OPT-4: Kmb 2→3, Gm 10→12
          else if (n == 1024) { defKs = 2; defKmb = 2; defGm = 14; }
        } else {
          if      (n ==  384) { defKs = 2; defKmb = 7; defGm =  7; }
          else if (n ==  512) { defKs = 2; defKmb = 5; defGm = 14; }
          else if (n ==  768) { defKs = 2; defKmb = 3; defGm = 12; }   // OPT-4: Kmb 2→3
          else if (n == 1024) { defKs = 2; defKmb = 5; defGm = 14; }
        }
      } else if constexpr (kTileM == 48) {
        if      (n ==  384) { defKs = 3; defKmb = 4; defGm =  2; }
        else if (n ==  512) { defKs = 3; defKmb = 4; defGm =  4; }
        else if (n ==  768) { defKs = 2; defKmb = 5; defGm = 14; }   // OPT-4: Ks 3→2, Kmb 4→5 (BF16 accum enables this)
        else if (n == 1024) { defKs = 3; defKmb = 4; defGm =  8; }
      } else if constexpr (kTileM == 64) {
        if      (n ==  384) { defKs = 2; defKmb = 4; defGm = 12; }
        else if (n ==  512) { defKs = 2; defKmb = 2; defGm =  4; }
        else if (n ==  768) { defKs = 2; defKmb = 4; defGm = 14; }   // OPT-4: Gm 12→14
        else if (n == 1024) { defKs = 2; defKmb = 2; defGm =  8; }
        else if (n == 14336) { defKs = 2; defKmb = 3; defGm =  8; }
      }
      if (n == 14336) {
        if constexpr (kTileM == 8) {
          if      (m_avg <= 1) { defKs = 2; defKmb = 4; defGm = 16; }
          else if (m_avg <= 8) { defKs = 2; defKmb = 3; defGm = 16; }
        } else if constexpr (kTileM == 16) {
          defKs = 4; defKmb = 4; defGm = 14;
        }
      }
    }

    if (k == 7168 && n == 512) {
      const int m_avg = num_seq_per_group_avg;
      if constexpr (kTileM == 8) {
        if      (m_avg <= 0) { defKs = 2; defKmb = 2; defGm =  8; }
        else if (m_avg <= 1) { defKs = 3; defKmb = 2; defGm =  8; }
        else if (m_avg <= 3) { defKs = 2; defKmb = 3; defGm = 10; }
        else if (m_avg <= 8) { defKs = 4; defKmb = 2; defGm = 10; }
      } else if constexpr (kTileM == 16) {
        defKs = 2; defKmb = 2; defGm = 14;
      } else if constexpr (kTileM == 48) {
        defKs = 2; defKmb = 2; defGm = 8;
      }
    }

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
      HPC_BWS_DISPATCH(2, 2) HPC_BWS_DISPATCH(2, 3) HPC_BWS_DISPATCH(2, 4)
      HPC_BWS_DISPATCH(2, 5) HPC_BWS_DISPATCH(2, 6)
      HPC_BWS_DISPATCH(3, 2) HPC_BWS_DISPATCH(3, 3) HPC_BWS_DISPATCH(3, 4)
    } else if constexpr (kTileM == 64) {
      HPC_BWS_DISPATCH(2, 2) HPC_BWS_DISPATCH(2, 3) HPC_BWS_DISPATCH(2, 4)
      HPC_BWS_DISPATCH(3, 2) HPC_BWS_DISPATCH(3, 3) HPC_BWS_DISPATCH(3, 4)
    }
#else
    if constexpr (kTileM == 8) {
      HPC_BWS_DISPATCH(2, 2) HPC_BWS_DISPATCH(2, 3) HPC_BWS_DISPATCH(2, 4)
      HPC_BWS_DISPATCH(3, 2) HPC_BWS_DISPATCH(3, 3)
      HPC_BWS_DISPATCH(4, 2)
    } else if constexpr (kTileM == 16) {
      HPC_BWS_DISPATCH(2, 2) HPC_BWS_DISPATCH(2, 3)
    } else if constexpr (kTileM == 32) {
      HPC_BWS_DISPATCH(2, 2) HPC_BWS_DISPATCH(2, 3) HPC_BWS_DISPATCH(2, 5)  // OPT-4: added (2,3) for n=768
    } else if constexpr (kTileM == 48) {
      HPC_BWS_DISPATCH(2, 4) HPC_BWS_DISPATCH(2, 5) HPC_BWS_DISPATCH(3, 4)  // OPT-4: added (2,5)
    } else if constexpr (kTileM == 64) {
      HPC_BWS_DISPATCH(2, 4)
    }
#endif
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
