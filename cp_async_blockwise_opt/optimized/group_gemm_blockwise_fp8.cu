// Copyright 2026 hpc-ops authors
//
// Blockwise FP8 group-GEMM (contiguous A) — cp.async + SS-WGMMA (SM90).
// >>> OPTIMIZED VERSION — targeting occupancy & bandwidth utilization <<<
//
// Changes vs original:
//   OPT-A: Eliminate tCS[kN] — fold xscale read into dequant FMA loop
//   OPT-B: Replace G2SCopyAS with inline cp.async (saves 4-6 registers)
//   OPT-C: Pre-read wscale before async fence (hide smem latency)
//   OPT-D: Add n=768 picker entries for Hunyuan-V3 TP=4 k=4096

#include <cuda.h>
#include <stdio.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "src/group_gemm/sm90/cp_async/common.cuh"
#include "src/group_gemm/sm90/cp_async/config.h"
#include "src/group_gemm/sm90/cp_async/group_gemm.h"
#include "src/utils/utils.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace group_gemm_cp_async {
namespace kernels {

template <typename Config, bool kUseTaskMap, bool kUsePDL, int kMinBlocks>
__global__ void __launch_bounds__(128, kMinBlocks)
    group_gemm_fp8_blockwise_multistage_kernel(
        const void *Cptr, const void *Aptr, const void *Bptr,
        const float *xscale_ptr, const float *wscale_ptr,
        const int *seqlens_ptr, const int *cu_seqlens_ptr,
        const int *tiles_ptr, const int *cu_tiles_ptr,
        const int4 *task_map_ptr, int task_map_len,
        int m, int n, int k, int num_group, int m_pad,
        int num_block_n, int num_block_k_pad4,
        cutlass::FastDivmod flat_divider) {
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
  // G2SCopyAS removed — replaced by inline-asm cp.async (OPT-B).
  using S2GCopyC = typename Config::S2GCopyC;
  using TiledMMA = typename Config::TiledMMA;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kStage = Config::kStage;
  constexpr int kMaxNtileK = Config::kMaxNtileK;
  constexpr int kWScaleNDiv = Config::kWScaleNDiv;

  static_assert(kTileK == 128,
                "blockwise FP8 cp.async kernel requires kTileK == 128");
  static_assert(kTileN == 64,
                "blockwise FP8 cp.async kernel currently assumes kTileN == 64");

  extern __shared__ uint8_t shm_data[];
  Tin *Ashm  = reinterpret_cast<Tin  *>(shm_data);
  Tin *Bshm  = Ashm + cosize(SmemLayoutA{});
  TS  *ASshm = reinterpret_cast<TS *>(Bshm + cosize(SmemLayoutB{}));
  TS  *WSshm = ASshm + cosize(SmemLayoutXS{});
  int *shm_tiles = reinterpret_cast<int *>(WSshm + cosize(SmemLayoutWS{}));
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

    Tensor A = make_tensor(
        make_gmem_ptr(reinterpret_cast<const Tin *>(Aptr)
                      + uint64_t(start_token) * k),
        make_shape(m_group, k), make_stride(k, Int<1>{}));
    Tensor B = make_tensor(
        make_gmem_ptr(reinterpret_cast<const Tin *>(Bptr)
                      + uint64_t(igroup) * n * k),
        make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor C = make_tensor(
        make_gmem_ptr(reinterpret_cast<Tout *>(const_cast<void *>(Cptr))
                      + uint64_t(start_token) * n),
        make_shape(n, m_group), make_stride(Int<1>{}, n));

    Tensor gA  = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}),
                            make_coord(itile_m, _));
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
    auto tgA = g2s_thr_a.partition_S(gA);
    auto tsA = g2s_thr_a.partition_D(sA);

    G2SCopyB g2s_tiled_copy_b;
    auto g2s_thr_b = g2s_tiled_copy_b.get_slice(idx);
    auto tgB = g2s_thr_b.partition_S(gB);
    auto tsB = g2s_thr_b.partition_D(sB);

    // ═══════════════════════════════════════════════════════════════
    // OPT-B: Replace G2SCopyAS CuTe TiledCopy with inline cp.async.
    //
    // The CuTe path's partition_S/D state (tgAS, tsAS, pred_xs)
    // consumed ~4-6 registers that survived the entire main loop.
    // Replace with two precomputed scalar pointers + an inline asm
    // lambda, saving registers at zero functional cost.
    // ═══════════════════════════════════════════════════════════════
    const bool xs_live = (idx < Config::kXsLiveThrs);
    const int  xs_thr_off = idx * 4;
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

    auto tIA = g2s_thr_a.partition_S(
        make_identity_tensor(make_shape(Int<kTileM>{}, Int<kTileK>{})));
    auto tIC = s2g_thr_c.partition_S(
        make_identity_tensor(make_shape(Int<kTileN>{}, Int<kTileM>{})));
    auto pred_a = make_tensor<bool>(shape(tIA));
    auto pred_c = make_tensor<bool>(shape(tIC));
#pragma unroll
    for (int i = 0; i < size(tIA); ++i) {
      pred_a(i) = get<0>(tIA(i)) < kTileM &&
                  itile_m * kTileM + get<0>(tIA(i)) < m_group;
    }
#pragma unroll
    for (int i = 0; i < size(tIC); ++i) {
      pred_c(i) = get<1>(tIC(i)) < kTileM &&
                  itile_m * kTileM + get<1>(tIC(i)) < m_group;
    }

    {
      constexpr int kWsVecs = kMaxNtileK / 4;
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
      cp_async_fence();
      cp_async_wait<0>();
      __syncthreads();
    }

    int itile_to_read = 0;
    int ismem_write = 0;
#pragma unroll
    for (int i = 0; i < kStage - 1; ++i) {
      if (i < ntile) {
        cute::copy_if(g2s_tiled_copy_a, pred_a,
                      tgA(_, _, _, i), tsA(_, _, _, i));
        cute::copy(g2s_tiled_copy_b,
                   tgB(_, _, _, i), tsB(_, _, _, i));
        if (xs_live) {
          xs_cp_async(/*itile=*/i, /*ismem=*/i);
        }
        ++itile_to_read;
        ++ismem_write;
      }
      cp_async_fence();
    }

    auto tDr = make_tensor_like(tCr);
    clear(tDr);

    // ───────── Main loop ─────────
    for (int itile = 0; itile < ntile; ++itile) {
      if (itile_to_read < ntile) {
        cute::copy_if(g2s_tiled_copy_a, pred_a,
                      tgA(_, _, _, itile_to_read),
                      tsA(_, _, _, ismem_write));
        cute::copy(g2s_tiled_copy_b,
                   tgB(_, _, _, itile_to_read),
                   tsB(_, _, _, ismem_write));
        if (xs_live) {
          xs_cp_async(/*itile=*/itile_to_read, /*ismem=*/ismem_write);
        }
        ++itile_to_read;
        ismem_write = (ismem_write + 1) % kStage;
      }
      cp_async_fence();
      cp_async_wait<kStage - 1>();
      __syncthreads();

      // OPT-C: pre-read wscale into register before the async fence.
      // sBS was fully loaded synchronously before the main loop, so
      // this generic-proxy ld.shared is safe after __syncthreads.
      // Reading it before fence_view_async_shared hides the ~20-cycle
      // smem latency behind the fence instruction.
      const float wscale_val = sBS(itile);

      cutlass::arch::fence_view_async_shared();

      const int ismem_read = itile % kStage;

      // ═══════════════════════════════════════════════════════════════
      // OPT-A: tCS[kN] ELIMINATED — scale read folded into dequant FMA.
      //
      // Original code pre-computed float tCS[kN] before WGMMA, keeping
      // kN FP32 registers live across the entire WGMMA sequence.
      // For kTileM=48 (kN≈12) that's 12 extra FP32 regs competing with
      // tCr+tDr for the register file, forcing ptxas to spill or lower
      // kMinBlocks.
      //
      // Fix: read xscale from sAS inline during the post-WGMMA FMA loop.
      // - kTileM ≤ 32: use __shfl_sync to broadcast from one lane that
      //   read the value from smem. Cost: ~5 cycles per N-column, fully
      //   hidden by FMA throughput.
      // - kTileM > 32: direct sAS smem read per N-column (~20 cycles),
      //   also hidden by interleaved FMA.
      //
      // Register savings: kN FP32 registers (8–16 depending on kTileM).
      // ═══════════════════════════════════════════════════════════════

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

      // Dequant FMA with inline xscale read (no tCS array).
      auto tDr_mn = retile_fragment(tDr);
      if constexpr (kTileM <= 32) {
        const int lane = idx & 31;
        const int my_m = lane % kTileM;
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

    cp_async_wait<0>();
    __syncthreads();

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

void group_gemm_fp8_blockwise_multistage_async(
    void *y_ptr, const void *x_ptr, const void *w_ptr,
    const void *xscale_ptr, const void *wscale_ptr,
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
         "blockwise cp.async: n must be a multiple of 64");
  assert(k % 128 == 0 &&
         "blockwise cp.async: k must be a multiple of 128");
  assert((k / 128) <= 128 &&
         "blockwise cp.async: k/128 (ntile_k) must be <= 128");

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
    const int shm_size =
        gemm_config.kShmSize + (num_group + 1) * sizeof(int);
    dim3 grid(get_sm_count() * gridMul);
    const bool use_task_map = (task_map_ptr != nullptr);

    auto dispatch = [&](auto use_task_map_tag, auto use_pdl_tag) {
      constexpr bool kUseTaskMap = decltype(use_task_map_tag)::value;
      constexpr bool kUsePDL = decltype(use_pdl_tag)::value;
      auto kernel =
          kernels::group_gemm_fp8_blockwise_multistage_kernel<GemmConfig,
                                                              kUseTaskMap,
                                                              kUsePDL,
                                                              kMinBlocks>;
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
            reinterpret_cast<const int *>(seqlens_ptr),
            reinterpret_cast<const int *>(cu_seqlens_ptr),
            reinterpret_cast<const int *>(tiles_ptr),
            reinterpret_cast<const int *>(cu_tiles_ptr),
            tm_ptr_typed, tm_ub, m, n, k, num_group,
            m_pad, num_block_n, num_block_k_pad4, flat_divider);
      }
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
  // OPT-D: Picker table with n=768 entries for Hunyuan-V3 TP=4
  // ═══════════════════════════════════════════════════════════════════
  auto pick_and_launch = [&](auto tile_m_tag) {
    constexpr int kTileM = decltype(tile_m_tag)::value;

    int defKs, defKmb, defGm;
    if constexpr (kTileM == 8) {
      defKs = 2; defKmb = 2; defGm = 9;
    } else if constexpr (kTileM == 16) {
      defKs = 2; defKmb = 2; defGm = 9;
    } else if constexpr (kTileM == 32) {
      defKs = 3; defKmb = 3; defGm = 5;
    } else if constexpr (kTileM == 48) {
      defKs = 2; defKmb = 3; defGm = 2;
    } else {
      defKs = 2; defKmb = 3; defGm = 3;
    }

    if (k == 4096) {
      const int m_avg = num_seq_per_group_avg;
      if constexpr (kTileM == 8) {
        if (m_avg <= 1) {
          if      (n ==  384) { defKs = 2; defKmb = 2; defGm =  9; }
          else if (n ==  512) { defKs = 4; defKmb = 2; defGm = 14; }
          else if (n ==  768) { defKs = 3; defKmb = 3; defGm = 14; }   // OPT-D: Hunyuan GEMM1
          else if (n == 1024) { defKs = 4; defKmb = 2; defGm = 14; }
        } else if (m_avg <= 2) {
          if      (n ==  384) { defKs = 2; defKmb = 2; defGm = 12; }
          else if (n ==  512) { defKs = 4; defKmb = 2; defGm =  4; }
          else if (n ==  768) { defKs = 3; defKmb = 3; defGm = 18; }   // OPT-D
          else if (n == 1024) { defKs = 4; defKmb = 2; defGm = 14; }
        } else if (m_avg <= 5) {
          if      (n ==  384) { defKs = 2; defKmb = 3; defGm = 15; }
          else if (n ==  512) { defKs = 4; defKmb = 2; defGm =  4; }
          else if (n ==  768) { defKs = 3; defKmb = 3; defGm = 18; }   // OPT-D
          else if (n == 1024) { defKs = 4; defKmb = 2; defGm = 14; }
        } else {
          if      (n ==  384) { defKs = 2; defKmb = 2; defGm =  9; }
          else if (n ==  512) { defKs = 4; defKmb = 2; defGm =  4; }
          else if (n ==  768) { defKs = 3; defKmb = 3; defGm = 18; }   // OPT-D
          else if (n == 1024) { defKs = 4; defKmb = 2; defGm = 14; }
        }
      } else if constexpr (kTileM == 16) {
        if (m_avg <= 10) {
          if      (n ==  384) { defKs = 2; defKmb = 2; defGm =  9; }
          else if (n ==  512) { defKs = 2; defKmb = 2; defGm =  7; }
          else if (n ==  768) { defKs = 2; defKmb = 3; defGm = 16; }   // OPT-D
          else if (n == 1024) { defKs = 4; defKmb = 4; defGm = 14; }
        } else {
          if      (n ==  384) { defKs = 2; defKmb = 2; defGm =  9; }
          else if (n ==  512) { defKs = 4; defKmb = 2; defGm =  4; }
          else if (n ==  768) { defKs = 2; defKmb = 3; defGm = 16; }   // OPT-D
          else if (n == 1024) { defKs = 4; defKmb = 2; defGm = 10; }
        }
      } else if constexpr (kTileM == 32) {
        if (m_avg <= 21) {
          if      (n ==  384) { defKs = 3; defKmb = 4; defGm =  4; }
          else if (n ==  512) { defKs = 3; defKmb = 3; defGm =  5; }
          else if (n ==  768) { defKs = 2; defKmb = 3; defGm = 10; }   // OPT-D
          else if (n == 1024) { defKs = 3; defKmb = 2; defGm = 14; }
        } else {
          if      (n ==  384) { defKs = 3; defKmb = 4; defGm = 10; }
          else if (n ==  512) { defKs = 3; defKmb = 3; defGm = 14; }
          else if (n ==  768) { defKs = 2; defKmb = 3; defGm = 10; }   // OPT-D
          else if (n == 1024) { defKs = 3; defKmb = 2; defGm = 14; }
        }
      } else if constexpr (kTileM == 48) {
        if      (n ==  384) { defKs = 2; defKmb = 3; defGm =  2; }
        else if (n ==  512) { defKs = 2; defKmb = 2; defGm = 14; }
        else if (n ==  768) { defKs = 2; defKmb = 3; defGm = 12; }   // OPT-D
        else if (n == 1024) { defKs = 2; defKmb = 2; defGm = 15; }
      } else if constexpr (kTileM == 64) {
        if      (n ==  384) { defKs = 2; defKmb = 3; defGm =  3; }
        else if (n ==  512) { defKs = 3; defKmb = 3; defGm =  4; }
        else if (n ==  768) { defKs = 2; defKmb = 3; defGm = 12; }   // OPT-D
        else if (n == 1024) { defKs = 3; defKmb = 3; defGm =  8; }
      }
    }

    if (k == 256) {
      const int m_avg = num_seq_per_group_avg;
      (void)m_avg;
      if constexpr (kTileM == 8) {
        defKs = 2; defKmb = 2; defGm = 8;
      } else if constexpr (kTileM == 16) {
        defKs = 2; defKmb = 2; defGm = 8;
      } else if constexpr (kTileM == 32) {
        defKs = 2; defKmb = 6; defGm = 14;
      } else if constexpr (kTileM == 48) {
        defKs = 2; defKmb = 5; defGm = 5;
      } else if constexpr (kTileM == 64) {
        defKs = 3; defKmb = 3; defGm = 4;
      }
    }

    if (k == 384) {
      if constexpr (kTileM == 8) {
        defKs = 3; defKmb = 3; defGm = 14;
      } else if constexpr (kTileM == 16) {
        defKs = 3; defKmb = 2; defGm = 14;
      } else if constexpr (kTileM == 32) {
      } else if constexpr (kTileM == 48) {
        defKs = 3; defKmb = 3; defGm = 10;
      } else if constexpr (kTileM == 64) {
        defKs = 3; defKmb = 3; defGm = 12;
      }
    }

    {
      char nm[40];
      const int tm = kTileM;
      std::snprintf(nm, sizeof(nm), "HPC_BW_KSTAGE_%d", tm);
      if (const char *s = std::getenv(nm)) defKs  = std::atoi(s);
      std::snprintf(nm, sizeof(nm), "HPC_BW_KMINBLK_%d", tm);
      if (const char *s = std::getenv(nm)) defKmb = std::atoi(s);
      std::snprintf(nm, sizeof(nm), "HPC_BW_GRIDMUL_%d", tm);
      if (const char *s = std::getenv(nm)) defGm  = std::atoi(s);
    }

#define HPC_BW_DISPATCH(KS, KMB)                                         \
    if (defKs == (KS) && defKmb == (KMB)) {                              \
      launch(Int<kTileM>{}, Int<(KS)>{}, Int<(KMB)>{}, defGm);           \
      return;                                                            \
    }

    if constexpr (kTileM == 8) {
      HPC_BW_DISPATCH(2, 2) HPC_BW_DISPATCH(2, 3) HPC_BW_DISPATCH(2, 4)
      HPC_BW_DISPATCH(2, 5) HPC_BW_DISPATCH(2, 6) HPC_BW_DISPATCH(2, 7)
      HPC_BW_DISPATCH(2, 8)
      HPC_BW_DISPATCH(3, 2) HPC_BW_DISPATCH(3, 3) HPC_BW_DISPATCH(3, 4)
      HPC_BW_DISPATCH(3, 5) HPC_BW_DISPATCH(3, 6)
      HPC_BW_DISPATCH(4, 2) HPC_BW_DISPATCH(4, 3) HPC_BW_DISPATCH(4, 4)
    } else if constexpr (kTileM == 16) {
      HPC_BW_DISPATCH(2, 2) HPC_BW_DISPATCH(2, 3) HPC_BW_DISPATCH(2, 4)
      HPC_BW_DISPATCH(2, 5) HPC_BW_DISPATCH(2, 6) HPC_BW_DISPATCH(2, 7)
      HPC_BW_DISPATCH(2, 8)
      HPC_BW_DISPATCH(3, 2) HPC_BW_DISPATCH(3, 3) HPC_BW_DISPATCH(3, 4)
      HPC_BW_DISPATCH(3, 5)
      HPC_BW_DISPATCH(4, 2) HPC_BW_DISPATCH(4, 3) HPC_BW_DISPATCH(4, 4)
    } else if constexpr (kTileM == 32) {
      HPC_BW_DISPATCH(2, 2) HPC_BW_DISPATCH(2, 3) HPC_BW_DISPATCH(2, 4)
      HPC_BW_DISPATCH(2, 5) HPC_BW_DISPATCH(2, 6) HPC_BW_DISPATCH(2, 7)
      HPC_BW_DISPATCH(3, 2) HPC_BW_DISPATCH(3, 3) HPC_BW_DISPATCH(3, 4)
      HPC_BW_DISPATCH(3, 5)
      HPC_BW_DISPATCH(4, 2) HPC_BW_DISPATCH(4, 3)
    } else if constexpr (kTileM == 48) {
      HPC_BW_DISPATCH(2, 2) HPC_BW_DISPATCH(2, 3) HPC_BW_DISPATCH(2, 4)
      HPC_BW_DISPATCH(2, 5)
      HPC_BW_DISPATCH(3, 2) HPC_BW_DISPATCH(3, 3) HPC_BW_DISPATCH(3, 4)
    } else if constexpr (kTileM == 64) {
      HPC_BW_DISPATCH(2, 2) HPC_BW_DISPATCH(2, 3) HPC_BW_DISPATCH(2, 4)
      HPC_BW_DISPATCH(3, 2) HPC_BW_DISPATCH(3, 3)
    }
#undef HPC_BW_DISPATCH

    assert(false &&
           "blockwise cp.async: picker produced an "
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
