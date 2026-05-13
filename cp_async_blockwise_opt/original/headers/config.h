// Copyright 2026 hpc-ops authors

#ifndef SRC_GROUP_GEMM_SM90_CP_ASYNC_CONFIG_H_
#define SRC_GROUP_GEMM_SM90_CP_ASYNC_CONFIG_H_

#include "cute/tensor.hpp"

namespace hpc {
namespace group_gemm_cp_async {

namespace config {

using namespace cute;  // NOLINT

template <int kTileM>
CUTE_HOST_DEVICE constexpr auto mma_selector() {
  if constexpr (kTileM == 8) {
    return SM90_64x8x32_F32E4M3E4M3_SS_TN<>{};
  } else if constexpr (kTileM == 16) {
    return SM90_64x16x32_F32E4M3E4M3_SS_TN<>{};
  } else if constexpr (kTileM == 32) {
    return SM90_64x32x32_F32E4M3E4M3_SS_TN<>{};
  } else if constexpr (kTileM == 48) {
    return SM90_64x48x32_F32E4M3E4M3_SS_TN<>{};
  } else if constexpr (kTileM == 64) {
    return SM90_64x64x32_F32E4M3E4M3_SS_TN<>{};
  } else {
    static_assert(kTileM == 8 || kTileM == 16 || kTileM == 32 || kTileM == 48 || kTileM == 64,
                  "mma_selector: kTileM must be one of {8, 16, 32, 48, 64}");
    return SM90_64x64x32_F32E4M3E4M3_SS_TN<>{};
  }
}

template <typename Tin_, typename Tout_, int kTileM_, int kTileN_, int kTileK_, int kStage_ = 2>
struct FP8GemmConfig {
  using Tin = Tin_;
  using Tout = Tout_;

  static constexpr int kTileM = kTileM_;
  static constexpr int kTileN = kTileN_;
  static constexpr int kTileK = kTileK_;
  static constexpr int kStage = kStage_;

  using WarpGroupLayout = decltype(make_layout(make_shape(Int<1>{}, Int<1>{}, Int<1>{})));
  using TiledMMA = decltype(make_tiled_mma(mma_selector<kTileM_>(), WarpGroupLayout{}));

  using SmemAtomA = std::conditional_t<kTileK_ >= 128, GMMA::Layout_K_SW128_Atom<Tin>,
                                       GMMA::Layout_K_SW64_Atom<Tin>>;
  using SmemAtomB = std::conditional_t<kTileK_ >= 128, GMMA::Layout_K_SW128_Atom<Tin>,
                                       GMMA::Layout_K_SW64_Atom<Tin>>;

  using SmemLayoutA =
      decltype(tile_to_shape(SmemAtomA{}, make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})));
  using SmemLayoutB =
      decltype(tile_to_shape(SmemAtomB{}, make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})));

  using SmemLayoutCT = decltype(tile_to_shape(GMMA::Layout_MN_SW64_Atom<Tout>{},
                                              make_shape(Int<kTileN>{}, Int<kTileM>{})));

  using CpAsyncAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Tin>;
  using G2SCopyA_FOR_TK_128 = decltype(make_tiled_copy(
      CpAsyncAtom{}, make_layout(make_shape(Int<16>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<16>{}), make_stride(Int<0>{}, Int<1>{}))));
  using G2SCopyA_FOR_TK_64 = decltype(make_tiled_copy(
      CpAsyncAtom{}, make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<16>{}), make_stride(Int<0>{}, Int<1>{}))));
  using G2SCopyA = std::conditional_t<kTileK_ >= 128, G2SCopyA_FOR_TK_128, G2SCopyA_FOR_TK_64>;
  using G2SCopyB = G2SCopyA;

  using R2SCopyAtomC = std::conditional_t<kTileM_ >= 16, Copy_Atom<SM90_U16x8_STSM_T, Tout>,
                                          Copy_Atom<SM90_U16x4_STSM_T, Tout>>;

  using S2GCopyC = decltype(make_tiled_copy(
      Copy_Atom<UniversalCopy<cute::uint128_t>, Tout>{},
      make_layout(make_shape(Int<8>{}, Int<16>{}), make_stride(Int<1>{}, Int<8>{})),
      make_layout(make_shape(Int<8>{}, Int<1>{}), make_stride(Int<1>{}, Int<0>{}))));

  static constexpr int shm_xw = sizeof(Tin) * (cosize(SmemLayoutA{}) + cosize(SmemLayoutB{}));
  static constexpr int shm_y = sizeof(Tout) * cosize(SmemLayoutCT{});
  static constexpr int kShmSize = shm_xw > shm_y ? shm_xw : shm_y;
};

template <int kTileM>
CUTE_HOST_DEVICE constexpr auto bw_mma_selector() {
  if constexpr (kTileM == 8) {
    return cute::SM90_64x8x32_F32E4M3E4M3_SS_TN<>{};
  } else if constexpr (kTileM == 16) {
    return cute::SM90_64x16x32_F32E4M3E4M3_SS_TN<>{};
  } else if constexpr (kTileM == 32) {
    return cute::SM90_64x32x32_F32E4M3E4M3_SS_TN<>{};
  } else if constexpr (kTileM == 48) {
    return cute::SM90_64x48x32_F32E4M3E4M3_SS_TN<>{};
  } else if constexpr (kTileM == 64) {
    return cute::SM90_64x64x32_F32E4M3E4M3_SS_TN<>{};
  } else {
    return cute::SM90_64x64x32_F32E4M3E4M3_SS_TN<>{};
  }
}

template <typename Tin_, typename Tout_, int kTileM_, int kTileN_, int kTileK_,
          int kStage_ = 4>
struct BlockWiseFP8GemmConfig {
  using Tin = Tin_;
  using Tout = Tout_;
  using TS = float;

  static constexpr int kTileM = kTileM_;
  static constexpr int kTileN = kTileN_;
  static constexpr int kTileK = kTileK_;
  static constexpr int kStage = kStage_;

  using WarpGroupLayout = decltype(make_layout(
      make_shape(Int<1>{}, Int<1>{}, Int<1>{})));
  using TiledMMA = decltype(make_tiled_mma(bw_mma_selector<kTileM_>(),
                                           WarpGroupLayout{}));

  using SmemAtomA = GMMA::Layout_K_SW128_Atom<Tin>;
  using SmemAtomB = GMMA::Layout_K_SW128_Atom<Tin>;
  using SmemLayoutA = decltype(tile_to_shape(
      SmemAtomA{}, make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemAtomB{}, make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})));
  using SmemLayoutCT = decltype(tile_to_shape(
      GMMA::Layout_MN_SW64_Atom<Tout>{},
      make_shape(Int<kTileN>{}, Int<kTileM>{})));

  using SmemLayoutXS = decltype(make_layout(
      make_shape (Int<kTileM>{}, Int<1>{}, Int<kStage>{}),
      make_stride(Int<1>{},      Int<0>{}, Int<kTileM>{})));

  static constexpr int kMaxNtileK = 128;
  using SmemLayoutWS = decltype(make_layout(
      make_shape(Int<kMaxNtileK>{}),
      make_stride(Int<1>{})));

  using CpAsyncAtom  = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Tin>;
  using CpAsyncAtomS = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, TS>;

  static_assert(kTileM_ % 4 == 0,
                "BlockWiseFP8GemmConfig: kTileM must be a multiple of 4");

  using G2SCopyA = decltype(make_tiled_copy(
      CpAsyncAtom{},
      make_layout(make_shape(Int<16>{}, Int<8>{}),
                  make_stride(Int<8>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<16>{}),
                  make_stride(Int<0>{}, Int<1>{}))));
  using G2SCopyB = G2SCopyA;

  static constexpr int kXsLiveThrs = kTileM_ / 4;
  using G2SCopyAS = decltype(make_tiled_copy(
      CpAsyncAtomS{},
      make_layout(make_shape (Int<kXsLiveThrs>{}, Int<1>{}),
                  make_stride(Int<1>{},           Int<0>{})),
      make_layout(make_shape (Int<4>{},           Int<1>{}))));

  static constexpr int kWScaleNDiv = 128 / kTileN_;

  using R2SCopyAtomC =
      std::conditional_t<kTileM_ >= 16,
                         Copy_Atom<SM90_U16x8_STSM_T, Tout>,
                         Copy_Atom<SM90_U16x4_STSM_T, Tout>>;

  using S2GCopyC = decltype(make_tiled_copy(
      Copy_Atom<UniversalCopy<cute::uint128_t>, Tout>{},
      make_layout(make_shape(Int<8>{}, Int<16>{}),
                  make_stride(Int<1>{}, Int<8>{})),
      make_layout(make_shape(Int<8>{}, Int<1>{}),
                  make_stride(Int<1>{}, Int<0>{}))));

  static constexpr int shm_xw = sizeof(Tin) *
      (cosize(SmemLayoutA{}) + cosize(SmemLayoutB{}));
  static constexpr int shm_xs = sizeof(TS) * cosize(SmemLayoutXS{});
  static constexpr int shm_ws = sizeof(TS) * cosize(SmemLayoutWS{});
  static constexpr int shm_y = sizeof(Tout) * cosize(SmemLayoutCT{});
  static constexpr int shm_used_during_loop = shm_xw + shm_xs + shm_ws;
  static constexpr int shm_used_during_epi  = shm_y  + shm_xs + shm_ws;
  static constexpr int kShmSize =
      (shm_used_during_loop > shm_used_during_epi) ? shm_used_during_loop
                                                   : shm_used_during_epi;

  auto get_shm_size() { return kShmSize; }
};

}  // namespace config
}  // namespace group_gemm_cp_async
}  // namespace hpc

#endif  // SRC_GROUP_GEMM_SM90_CP_ASYNC_CONFIG_H_
