# Blockwise Kernel 寄存器消耗分析与降低策略

## 1. 寄存器消耗分解

SS-WGMMA m64×N×k32 (FP8→FP32) 的输出分布: 每个 128 线程的 warpgroup 产出 64×N 个 FP32，
每线程持有 `64 × N / 128 = N/2` 个 FP32 累加器。

### 按 kTileM 分解 (单位: FP32 寄存器/线程)

| 组件 | kTileM=8 | kTileM=16 | kTileM=32 | kTileM=48 | kTileM=64 |
|------|----------|-----------|-----------|-----------|-----------|
| **tCr** (WGMMA 累加器) | 4 | 8 | 16 | 24 | 32 |
| **tDr** (dequant 累加器) | 4 | 8 | 16 | 24 | 32 |
| **tCS[kN]** (scale 预计算) | 4 | 8 | 8 | 12 | 16 |
| G2SCopyA 分区状态 (tgA+tsA) | ~6 | ~6 | ~6 | ~6 | ~6 |
| G2SCopyB 分区状态 (tgB+tsB) | ~6 | ~6 | ~6 | ~6 | ~6 |
| G2SCopyAS 分区状态 (tgAS+tsAS+pred_xs) | ~5 | ~5 | ~5 | ~5 | ~5 |
| S2GCopyC 分区状态 (tCs2g+tCg2s) | ~4 | ~4 | ~4 | ~4 | ~4 |
| pred_a + pred_c (谓词) | ~2 | ~2 | ~4 | ~4 | ~6 |
| gI / tI_mn (identity 坐标) | ~4 | ~4 | ~4 | ~4 | ~4 |
| 地址与偏移 (Aptr, Bptr, ws_base, ...) | ~10 | ~10 | ~10 | ~10 | ~10 |
| 循环变量 (itile, ismem_write, ...) | ~6 | ~6 | ~6 | ~6 | ~6 |
| **总计 (估算)** | **~55** | **~67** | **~85** | **~105** | **~127** |

### 关键观察

1. **tCr + tDr 占主导**: kTileM=48 时累加器占 48/105 = 46%，kTileM=64 时占 64/127 = 50%
2. **tCS[kN] 是次要但可消除的**: kTileM=48 时 12 个寄存器跨越 WGMMA 存活（已由 OPT-A 消除）
3. **CuTe 分区状态是固定开销**: ~21 个寄存器来自 G2SCopy{A,B,AS} + S2GCopyC，与 kTileM 无关
4. **SM90 寄存器预算**: 65536 regs/SM ÷ 128 threads/block：
   - kMinBlocks=2 → 256 regs/thread（充裕）
   - kMinBlocks=3 → 170 regs/thread（kTileM≥48 紧张）
   - kMinBlocks=4 → 128 regs/thread（kTileM≥32 紧张）

## 2. 已应用的优化 (OPT-A, OPT-B, OPT-3)

| 优化 | 节省 | 适用范围 | 原理 |
|------|------|---------|------|
| OPT-A: 消除 tCS[kN] | 8–16 regs | 所有 kTileM | scale 读取内联到 FMA 循环 |
| OPT-B: 内联 cp.async 替代 G2SCopyAS | 2–3 regs | 所有 kTileM | 删除 CuTe partition 元数据 |
| OPT-3: BF16 tDr | 12–16 regs | kTileM≥48 | FP32→BF16 累加器 |

优化后状态:

| kTileM | 原始 | OPT-A | +OPT-B | +OPT-3 | 可达 kMinBlocks |
|--------|------|-------|--------|--------|----------------|
| 8      | ~55  | ~51   | ~49    | —      | 5+ |
| 16     | ~67  | ~59   | ~57    | —      | 4 |
| 32     | ~85  | ~77   | ~75    | —      | 3 |
| 48     | ~105 | ~93   | ~91    | ~79    | 3 |
| 64     | ~127 | ~111  | ~109   | ~93    | 2–3 |

## 3. 进一步降低策略

### 策略 A: 内联 cp.async 替代 G2SCopyA / G2SCopyB

**节省: ~6–10 regs** | **复杂度: 高** | **风险: 中**

原始的 CuTe `G2SCopyA` 创建了多层 partition 对象:
```cpp
G2SCopyA g2s_tiled_copy_a;
auto g2s_thr_a = g2s_tiled_copy_a.get_slice(idx);
auto tgA = g2s_thr_a.partition_S(gA);  // gmem 分区 → 含 pointer + stride + offset 状态
auto tsA = g2s_thr_a.partition_D(sA);  // smem 分区 → 含 pointer + stride + offset 状态
```

每个分区对象在 CuTe 内部是一个 `Tensor<pointer, Layout<Shape, Stride>>`:
- 指针基地址: 1 reg (64-bit → 2 regs)
- Shape/Stride 的运行时部分: 2–4 regs (取决于 rank)
- tgA 和 tsA 合计: ~6 regs (跨越整个 main loop)

**用手写 cp.async 替代:**

```cpp
// 替代 CuTe copy_if(g2s_copy_a, pred_a, tgA(_, _, _, itile), tsA(_, _, _, ismem))
// Config: kTileM=8, kTileK=128, ThrLayout=(16,8), ValLayout=(1,16)
// 每线程负责 1 行, 16 字节 (128-bit cp.async)
__device__ __forceinline__ void load_A_tile_inline(
    void *smem_base,         // tsA 对应的 smem 起始 (考虑 swizzle)
    const void *gmem_base,   // gA 对应的 gmem 起始
    int K,                   // stride
    int num_valid_rows,      // M 方向有效行数
    int itile_k,             // K tile index
    int ismem                // smem stage index
) {
    constexpr int kTileK = 128;
    constexpr int kBytesPerAtom = 16;
    constexpr int kThreadsPerRow = kTileK / kBytesPerAtom;  // 8
    const int row = threadIdx.x / kThreadsPerRow;  // 哪一行
    const int col = threadIdx.x % kThreadsPerRow;  // 行内偏移

    // smem 地址: 需要考虑 SW128 swizzle pattern
    // gmem 地址: row * K + itile_k * kTileK + col * 16
    uint32_t smem_addr = /* 用 swizzle 函数计算 */;
    const void *gmem_ptr = (const char*)gmem_base + row * K + itile_k * kTileK + col * 16;
    int src_size = (row < num_valid_rows) ? 16 : 0;

    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n"
        :: "r"(smem_addr), "l"(gmem_ptr), "n"(16), "r"(src_size));
}
```

**难点**: SW128 swizzle pattern 的地址计算。CuTe 的 `Layout_K_SW128_Atom` 使用
128 字节的 swizzle，手写需要正确复现:

```cpp
// SW128 swizzle for kTileK=128:
// 每 128 字节一个 swizzle group。对于 FP8 (1 byte):
// bank = (byte_offset / 16) % 8
// row 对 bank 做 XOR: effective_bank = bank ^ (row % 8)
// 这会打散 bank conflicts
static __device__ uint32_t sw128_smem_addr(
    void *base, int row, int col_atom, int stage, int tile_m, int tile_k, int num_stage) {
    int offset_in_stage = row * tile_k + col_atom * 16;
    // SW128: XOR the top 3 bits of the 16B-atom index with the row bits
    int atom_idx = offset_in_stage / 16;
    int swizzled_atom = atom_idx ^ ((row & 7) << 0);  // 简化; 实际 pattern 见 CuTe 源码
    int byte_offset = swizzled_atom * 16 + stage * tile_m * tile_k;
    return cast_smem_ptr_to_uint((char*)base + byte_offset);
}
```

**实际建议**: 只对 kTileM ≥ 48 做此优化（收益最大），并用 CuTe 的 `swizzle` API
在 host 侧预计算 swizzle 参数传给 kernel。

---

### 策略 B: 消除 tDr — 分块累加到 SMEM

**节省: 24–32 regs (kTileM=48/64)** | **复杂度: 中** | **风险: 低**

最激进的寄存器节省策略。核心思想: 不在寄存器里保持完整的 tDr，而是
周期性地把 partial sum 写入 SMEM。

```
原始流程:
  for itile in 0..ntile:
    WGMMA → tCr
    tDr += tCr * yscale      ← tDr 全程存活 (24-32 regs)
  epilogue: convert tDr → BF16 output

分块流程 (CHUNK=8):
  for chunk in 0..ntile/CHUNK:
    for itile in chunk*CHUNK..(chunk+1)*CHUNK:
      WGMMA → tCr
      tCr *= yscale           ← 就地缩放 (无 tDr!)
      tCr_accum += tCr        ← tCr 自身做块内累加 (用 ScaleOut::One 部分替代)
    store tCr_accum → smem_partial[chunk]   ← spill
    clear tCr_accum (compiler 可复用这些 regs)
  
  // 最终 reduce
  load all smem_partial → tCr
  tCr = sum of all partials
  epilogue: convert tCr → BF16 output
```

**Wait** — 这还是不行。因为 WGMMA 的 `ScaleOut::Zero` 会覆盖 tCr，
所以块内不能同时用 tCr 做 WGMMA 输出和累加。

**正确版本**: 块内仍然用 tDr，但 tDr 的生命周期从 ntile 次迭代缩短到 CHUNK 次。
每 CHUNK 次迭代后，把 tDr 加到 smem 里的 running sum，然后 clear tDr。

```cpp
// 实际代码结构
constexpr int CHUNK = 8;  // 每 8 个 K-tile spill 一次
float *smem_partial = reinterpret_cast<float *>(/* 复用 sA 区域 */);
// smem_partial 大小: kM * kN * sizeof(float) per thread
// kTileM=48: 24 floats/thread × 4 = 96 bytes/thread × 128 threads = 12KB

clear(tDr);
for (int itile = 0; itile < ntile; ++itile) {
    // ... cp.async, WGMMA, dequant into tDr (unchanged) ...

    if ((itile + 1) % CHUNK == 0 || itile == ntile - 1) {
        // Spill tDr to smem and clear
        #pragma unroll
        for (int i = 0; i < size(tDr); ++i) {
            // 使用 atomicAdd 或 plain store (只有本线程访问自己的 slot)
            smem_partial[threadIdx.x * kAccumSize + i] += tDr(i);
        }
        clear(tDr);
    }
}

// Final reload
#pragma unroll
for (int i = 0; i < size(tDr); ++i) {
    tDr(i) = smem_partial[threadIdx.x * kAccumSize + i];
}
// ... epilogue ...
```

**编译器行为**: ptxas 看到 tDr 在 `clear(tDr)` 后"死亡"，会在下一次 dequant
使用前重新分配（可能是相同物理寄存器）。关键是 spill 代码在 K-loop 的
**迭代边界** 打断了 tDr 的活跃范围，让 ptxas 有机会压缩寄存器。

**实际效果**: 对 ptxas 的寄存器分配器来说，缩短活跃范围的主要好处在于：
原本 tDr 跨越 `cp.async + fence + wait + sync + WGMMA + dequant` 全程存活，
与 cp.async 发射期间的地址计算寄存器**重叠**。分块后在 spill 边界打断，
让地址计算和 tDr 可以分时复用。

**代价**: 每 CHUNK 次多 2 × 24 = 48 次 smem 操作 (st.shared + ld.shared for reload)。
对于 ntile=32, CHUNK=8: 4 次 spill × 48 ops × ~4 cycles = ~768 cycles ≈ 0.5 μs。

---

### 策略 C: 延迟 epilogue 分区创建

**节省: ~4 regs** | **复杂度: 低** | **风险: 无**

原始代码在 main loop 之前就创建了 S2GCopyC 的分区状态:

```cpp
S2GCopyC s2g_tiled_copy_c;
auto s2g_thr_c = s2g_tiled_copy_c.get_slice(idx);
auto tCs2g = s2g_thr_c.partition_S(sCT);   // ← 在 main loop 之前, 跨越整个 loop
auto tCg2s = s2g_thr_c.partition_D(gCT);   // ← 同上
```

`tCs2g` 和 `tCg2s` 只在 epilogue 使用，但它们的寄存器从 main loop 开始就被占用。

**修复**: 把 S2GCopyC 的所有代码移到 epilogue 内:

```cpp
// AFTER main loop, BEFORE epilogue:
cp_async_wait<0>();
__syncthreads();

// 现在才创建 epilogue copy state
S2GCopyC s2g_tiled_copy_c;
auto s2g_thr_c = s2g_tiled_copy_c.get_slice(idx);
auto tCs2g = s2g_thr_c.partition_S(sCT);
auto tCg2s = s2g_thr_c.partition_D(gCT);
// pred_c 也移到这里
```

ptxas 看到这些变量在 main loop 中不存在，可以将其寄存器用于 loop 内其他用途。

**注意**: pred_c 需要 `itile_m` 和 `m_group`，这些值在 while 循环内有效。
只需确保它们仍可访问。

---

### 策略 D: 手写 tI_mn 坐标表替代 identity tensor

**节省: ~2–4 regs** | **复杂度: 低** | **风险: 无**

```cpp
// 原始: CuTe identity tensor → 4 层对象
auto gI = make_identity_tensor(gCT.shape());
auto tI = thr_mma.partition_C(gI);
auto tI_mn = retile_fragment(tI);
// 使用: get<1>(tI_mn(0, in)) → 获取第 in 列的 m 坐标
```

CuTe 的 identity tensor 会保留 shape/stride 元数据在寄存器中。
但 `get<1>(tI_mn(0, in))` 的结果在编译时就可以确定（固定的 MMA 分区模式）。

**替代**: 用 constexpr 数组:

```cpp
// 预计算 m_idx 表 (编译时常量, 零寄存器)
// 这些值取决于 WGMMA 的输出分布 pattern, 对于 SS_TN 是固定的
constexpr int m_idx_table[kN] = { /* 从 CuTe identity tensor dump 出的值 */ };

// dequant loop:
for (int in = 0; in < kN; ++in) {
    const float yscale = sAS(m_idx_table[in], 0, ismem_read) * wscale_val;
    // ...
}
```

**如何获取值**: 写一个 host-side utility 打印 tI_mn 的内容:

```cpp
// Debug kernel (一次性运行, 提取坐标表)
if (threadIdx.x == 0 && blockIdx.x == 0) {
    for (int in = 0; in < kN; ++in) {
        printf("m_idx[%d] = %d\n", in, (int)get<1>(tI_mn(0, in)));
    }
}
```

---

### 策略 E: 编译时分支消除 (full tile 特化)

**节省: ~2–4 regs** | **复杂度: 低** | **风险: 无**

pred_a 只有在 partial tile (itile_m * kTileM + row >= m_group) 时才非全 true。
对于大多数 tile, `is_full_tile = true` → pred_a 全为 true → `copy_if` 等价于 `copy`。

在 while(true) 循环体内将 full tile 和 partial tile 分成两个代码路径:

```cpp
if (is_full_tile) {
    // 不需要 pred_a → 编译器可以消除 pred_a 的寄存器
    // 使用 cute::copy 替代 cute::copy_if
    run_main_loop</*kFullTile=*/true>(...);
} else {
    run_main_loop</*kFullTile=*/false>(...);
}
```

这让 full-tile 路径的 pred_a 完全被 DCE(dead code elimination)，释放对应寄存器。
代价是代码膨胀（两份 main loop 实例），但 icache 影响通常可忽略。

---

## 4. 策略优先级总结

按 **收益/复杂度** 排序:

| 优先级 | 策略 | 节省 (kTileM=48) | 复杂度 | 状态 |
|--------|------|-----------------|--------|------|
| ★★★★★ | OPT-A: 消除 tCS[kN] | 12 regs | 低 | ✅ 已实现 |
| ★★★★☆ | OPT-B: 内联 xscale cp.async | 3 regs | 低 | ✅ 已实现 |
| ★★★★☆ | OPT-3: BF16 tDr | 12 regs | 低 | ✅ 已实现 (scatter) |
| ★★★☆☆ | 策略 C: 延迟 epilogue 分区 | 4 regs | 低 | 🔲 待实现 |
| ★★★☆☆ | 策略 D: constexpr m_idx 表 | 2–4 regs | 低 | 🔲 待实现 |
| ★★★☆☆ | 策略 E: full-tile 特化 | 2–4 regs | 低 | 🔲 待实现 |
| ★★☆☆☆ | 策略 A: 内联 G2SCopy{A,B} | 6–10 regs | 高 | 🔲 需 swizzle |
| ★★☆☆☆ | 策略 B: 分块 spill tDr | 24 regs | 中 | 🔲 需验证 |

### 组合效果预估 (kTileM=48)

```
原始:                                  ~105 regs → kMinBlocks=2
+ OPT-A (消除 tCS):                    ~93 regs
+ OPT-B (内联 xscale):                 ~91 regs
+ OPT-3 (BF16 tDr):                   ~79 regs
+ 策略 C (延迟 epilogue):              ~75 regs
+ 策略 D (constexpr m_idx):            ~73 regs
+ 策略 E (full-tile DCE):              ~71 regs  → kMinBlocks=3 (170 budget)
+ 策略 A (内联 G2SCopy A/B):           ~63 regs  → kMinBlocks=4 (128 budget) ← 可能
```

## 5. 验证方法

```bash
# 编译时检查寄存器数
nvcc -Xptxas -v ... 2>&1 | grep "Used .* registers"

# 或用 cuobjdump
cuobjdump -res-usage your_kernel.cubin

# NCU 运行时检查
ncu --metrics launch__registers_per_thread,\
              l1tex__data_pipe_lsu_wavefronts_mem_local_op_ld.sum,\
              l1tex__data_pipe_lsu_wavefronts_mem_local_op_st.sum \
    python bench.py

# local_op_ld/st > 0 → 发生了寄存器 spill!
# 此时 ptxas 选择 spill 到 local memory (L1 cache backing),
# 每次 spill 增加 ~10-30 cycles 延迟
```
