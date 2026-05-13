# Blockwise cp_async Group-GEMM 性能分析指南

## 1. 理论 Roofline 建模

### 1.1 每个 expert tile 的算术强度

对于单个 tile (kTileM × kTileN × kTileK):

```
FLOPs = 2 × kTileM × kTileN × kTileK   (FP8 matmul)
      + 2 × kTileM × kTileN             (dequant FMA: scale + accumulate)

Bytes_loaded =
  A tile:     kTileM × kTileK × sizeof(fp8)  = kTileM × 128 bytes
  B tile:     kTileN × kTileK × sizeof(fp8)  = 64 × 128 = 8192 bytes
  xscale:     kTileM × sizeof(fp32)          = kTileM × 4 bytes
  wscale:     1 × sizeof(fp32)               = 4 bytes (amortized)

Bytes_stored =
  C tile:     kTileM × kTileN × sizeof(bf16) = kTileM × 128 bytes
  (only at epilogue, amortized over ntile K-iterations)
```

**Hunyuan-V3 TP=4 关键形状 (k=4096, n=768):**

| BS | tokens/expert | kTileM | ntile_k | A_bytes/tile | B_bytes/tile | FLOPs/tile | AI (FLOPs/Byte) |
|----|---------------|--------|---------|-------------|-------------|------------|-----------------|
| 32 | ~1.3 | 8 | 32 | 1024 | 8192 | 131,072+1024 | **14.3** |
| 128| ~5.3 | 8 | 32 | 1024 | 8192 | 131,072+1024 | **14.3** |
| 512| ~21 | 32 | 32 | 4096 | 8192 | 524,288+4096 | **42.9** |
| 1024| ~42 | 48 | 32 | 6144 | 8192 | 786,432+6144 | **55.0** |

H100 SXM: peak FP8 = 1979 TFLOPS, HBM BW = 3.35 TB/s
→ **Roofline 拐点** = 1979/3.35 ≈ **590 FLOPs/Byte**

**所有 Hunyuan 形状的 AI 远低于拐点 → 这是一个 memory-bound kernel。**

但需要注意：B 矩阵跨 expert 有 L2 复用机会；scatter A 完全随机访问。

### 1.2 有效带宽计算

从端到端 benchmark 数据反推实际带宽利用率：

```python
def effective_bandwidth(latency_us, bs, num_expert, topk, hidden, n_tp):
    """计算 fuse_moe 的有效 HBM 带宽."""
    tokens = bs * topk  # 总 token 数
    m_avg = tokens / num_expert

    # GEMM1 (gate+up): n=n_tp*2, k=hidden
    n1, k1 = n_tp * 2, hidden
    # GEMM2 (down): n=hidden, k=n_tp (简化; 实际取决于 activation 后的维度)
    n2, k2 = hidden, n_tp

    # 每 expert 数据量 (bytes)
    bytes_per_expert = (
        m_avg * k1 * 1       # A (fp8) for GEMM1
        + n1 * k1 * 1        # B (fp8) for GEMM1 (但多 expert 共享 L2)
        + m_avg * k2 * 1     # A (fp8) for GEMM2
        + n2 * k2 * 1        # B (fp8) for GEMM2
        + m_avg * n1 * 2     # C (bf16) GEMM1 output
        + m_avg * n2 * 2     # C (bf16) GEMM2 output
        # scales (small, ignored)
    )
    total_bytes = bytes_per_expert * num_expert
    bw_TB_s = total_bytes / (latency_us * 1e-6) / 1e12
    return bw_TB_s, total_bytes

# 示例: BS=128, bws_cpa = 337.57 μs
bw, total = effective_bandwidth(337.57, 128, 192, 8, 4096, 384)
print(f"Effective BW: {bw:.2f} TB/s (of 3.35 TB/s = {bw/3.35*100:.1f}%)")
```

## 2. NCU 性能剖析

### 2.1 关键命令

```bash
# 基础 metrics (占用率 + 计算/内存利用率)
ncu --target-processes all \
    --metrics \
    sm__warps_active.avg.pct_of_peak_sustained_active,\
    sm__cycles_elapsed.avg,\
    l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
    launch__registers_per_thread,\
    launch__occupancy \
    python bench_script.py

# 详细 memory 分析
ncu --target-processes all \
    --metrics \
    dram__bytes_read.sum,\
    dram__bytes_write.sum,\
    lts__throughput.avg.pct_of_peak_sustained_elapsed,\
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,\
    sm__sass_data_bytes_mem_shared_op_ld.sum,\
    sm__sass_data_bytes_mem_shared_op_st.sum \
    python bench_script.py

# cp.async 流水线分析
ncu --target-processes all \
    --metrics \
    sm__mio_pq_read_cycles_active.avg,\
    sm__mio_pq_write_cycles_active.avg,\
    l1tex__m_xbar2l1_read_bytes.sum \
    python bench_script.py

# 寄存器压力与 spill 分析
ncu --target-processes all \
    --metrics \
    launch__registers_per_thread,\
    launch__occupancy,\
    l1tex__data_pipe_lsu_wavefronts_mem_local_op_ld.sum,\
    l1tex__data_pipe_lsu_wavefronts_mem_local_op_st.sum \
    python bench_script.py
```

### 2.2 关键指标解读

| 指标 | 目标值 | 含义 |
|------|--------|------|
| `launch__registers_per_thread` | ≤128 (kTileM=8), ≤170 (kTileM=48) | 每线程寄存器数。过高→降低 occupancy |
| `launch__occupancy` | ≥50% (kTileM=8), ≥25% (kTileM=48) | 理论 occupancy |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | ≥40% | 实际活跃 warp 占比 |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | ≥70% | HBM 带宽利用率 |
| `lts__throughput.avg.pct_of_peak_sustained_elapsed` | 观察 | L2 吞吐 (scatter A 的缓存效果) |
| `sm__pipe_tensor_op_hmma_cycles_active` | 观察 | WGMMA 计算利用率 |
| `local_op_ld/st` | **= 0** | 寄存器溢出到 local memory → 性能杀手 |

### 2.3 分辨 memory-bound vs compute-bound

```
if dram_throughput > 70% AND tensor_op_utilization < 30%:
    → memory-bound (大多数 small-BS 情况)
    → 优化方向: 提高 occupancy, L2 缓存命中, 减少 scatter 开销

if tensor_op_utilization > 50% AND dram_throughput < 50%:
    → compute-bound (大 BS, kTileM=48/64)
    → 优化方向: 减少 dequant FMA 开销, 提高 WGMMA 利用率

if both < 40%:
    → latency-bound (pipeline stall)
    → 优化方向: 增加 pipeline 深度, 提高 occupancy 隐藏延迟
```

## 3. 分组分析法 — 逐层剥离瓶颈

### 3.1 Ablation 测试 (利用已有的 kAblMask)

scatter kernel 已经内置了 ablation 机制：

```bash
# mask=0: 完整 kernel (baseline)
HPC_BWS_ABLATE=0 python bench.py

# mask=1: 跳过 xscale 加载 → 量化 xscale cp.async 的开销
HPC_BWS_ABLATE=1 python bench.py

# mask=2: 跳过 wscale 加载 → 量化 wscale 预加载的开销
HPC_BWS_ABLATE=2 python bench.py

# mask=4: 跳过 dequant FMA → 量化 WGMMA 后缩放计算的开销
HPC_BWS_ABLATE=4 python bench.py

# mask=7: 跳过所有 scale 操作 → 纯 group-GEMM 的理论下限
HPC_BWS_ABLATE=7 python bench.py
```

通过对比: `mask=0 - mask=4` = dequant FMA 的纯开销。

### 3.2 逐参数扫描

用环境变量 override picker 参数，扫描 (kStage, kMinBlocks, gridMul):

```bash
# 扫描 kStage 对 kTileM=8 的影响
for KS in 2 3 4; do
  for KMB in 2 3 4; do
    for GM in 8 10 12 14 16 18 20; do
      HPC_BWS_KSTAGE_8=$KS HPC_BWS_KMINBLK_8=$KMB HPC_BWS_GRIDMUL_8=$GM \
        python bench_single_shape.py --bs=128 --variant=bws_cpa 2>/dev/null
    done
  done
done | tee sweep_results.txt
```

### 3.3 SMEM 占用量计算

从 `config.h` 的 `BlockWiseFP8GemmConfig` 可以精确计算:

```python
def smem_bytes(tile_m, tile_n, tile_k, stage):
    """计算 blockwise kernel 的 SMEM 占用."""
    # SW128 atom for fp8: cosize = tile_dim * tile_k (with swizzle padding)
    # Approximate: atom covers (M, 128) with 128B swizzle
    sA = tile_m * tile_k * stage       # fp8, 1 byte each
    sB = tile_n * tile_k * stage       # fp8, 1 byte each
    sAS = tile_m * 1 * stage * 4       # fp32 xscale
    sBS = 128 * 4                      # fp32 wscale (kMaxNtileK=128)
    # C epilogue overlaps A region
    sC = tile_n * tile_m * 2           # bf16
    compute_phase = sA + sB + sAS + sBS
    epilogue_phase = sC + sAS + sBS    # C overlaps A
    return max(compute_phase, epilogue_phase)

# 示例
for stage in [2, 3, 4]:
    s = smem_bytes(8, 64, 128, stage)
    max_blocks = 228 * 1024 // s
    print(f"kStage={stage}: {s/1024:.1f} KB → max {max_blocks} blocks/SM")
```

输出:
```
kStage=2: ~18.5 KB → max 12 blocks/SM
kStage=3: ~27.0 KB → max 8 blocks/SM
kStage=4: ~35.5 KB → max 6 blocks/SM
```

## 4. 针对本 kernel 的特殊分析维度

### 4.1 Scatter A 的 L2 缓存分析

scatter kernel 的 A 矩阵通过 `row_indices` 随机访问，L2 命中率直接影响性能:

```bash
ncu --metrics \
    lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum,\
    lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum,\
    lts__t_sectors_srcunit_tex_op_read.sum \
    python bench.py
```

```
L2_hit_rate = hit / (hit + miss)
```

- hit_rate > 80%: A 行被多 tile 复用 (同 expert 的不同 N-tile)
- hit_rate < 50%: scatter 模式严重破坏 L2 → 考虑 prefetch 优化 (OPT-1)

### 4.2 Expert 负载均衡

group-GEMM 的性能高度依赖 expert 间的负载均衡:

```python
import torch
topk_ids = torch.randint(0, num_expert, (bs, topk))
tokens_per_expert = torch.bincount(topk_ids.flatten(), minlength=num_expert)
print(f"min={tokens_per_expert.min()}, max={tokens_per_expert.max()}, "
      f"mean={tokens_per_expert.float().mean():.1f}, "
      f"std={tokens_per_expert.float().std():.1f}")
# 如果 max/mean > 3x → 严重不均衡, persistent block scheduler 效率下降
```

### 4.3 Wave efficiency (grid 利用率)

```python
def wave_efficiency(num_tiles_total, num_sm, grid_mul):
    """计算 persistent grid 的 wave 效率."""
    grid_size = num_sm * grid_mul
    num_waves = (num_tiles_total + grid_size - 1) // grid_size
    active_in_last_wave = num_tiles_total % grid_size
    if active_in_last_wave == 0:
        active_in_last_wave = grid_size
    efficiency = (num_tiles_total) / (num_waves * grid_size)
    return efficiency, num_waves

# Hunyuan BS=32: ~192 experts × 1 tile_m × 12 tile_n = 2304 total tiles
eff, waves = wave_efficiency(2304, 132, 10)
print(f"Wave efficiency: {eff:.1%}, waves: {waves}")
```

### 4.4 Pipeline bubble 分析

```
理想流水线 (kStage=3):

  Stage 0:  [cp.async A₂,B₂]  [WGMMA₀]         [dequant₀]
  Stage 1:  [cp.async A₃,B₃]  [WGMMA₁]         [dequant₁]
  Stage 2:  [cp.async A₄,B₄]  [WGMMA₂]         [dequant₂]
  
  如果 cp.async latency > WGMMA + dequant 时间 → pipeline bubble
  如果 WGMMA + dequant > cp.async latency → compute-bound (good)
```

用 NCU 的 warp stall 分析:

```bash
ncu --metrics \
    smsp__warps_issue_stalled_wait_barrier.avg,\
    smsp__warps_issue_stalled_membar.avg,\
    smsp__warps_issue_stalled_long_scoreboard.avg,\
    smsp__warps_issue_stalled_short_scoreboard.avg \
    python bench.py
```

- `stalled_long_scoreboard` 高 → 等待 global memory (cp.async 未完成)
- `stalled_wait_barrier` 高 → __syncthreads 导致的 bubble
- `stalled_short_scoreboard` 高 → 等待 smem 操作或 WGMMA

## 5. 优化前后对比清单

```bash
#!/bin/bash
# 完整对比脚本

METRICS="launch__registers_per_thread,\
launch__occupancy,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
l1tex__data_pipe_lsu_wavefronts_mem_local_op_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_local_op_st.sum"

echo "=== BEFORE (original kernel) ==="
ncu --metrics $METRICS python bench_original.py 2>&1 | tee ncu_before.txt

echo "=== AFTER (optimized kernel) ==="
ncu --metrics $METRICS python bench_optimized.py 2>&1 | tee ncu_after.txt

# 关键对比项:
# 1. registers_per_thread: 应该下降 (OPT-A 消除 tCS, OPT-B 消除 CuTe 状态)
# 2. occupancy: 应该上升 (更低 smem + 更低 register)
# 3. local_op_ld/st: 应该从 >0 变为 0 (消除 spill)
# 4. dram_throughput: 应该上升 (更多活跃 warp 维持带宽)
# 5. tensor_op: 可能上升 (减少 pipeline bubble)
```

## 6. 快速 Benchmark 脚本 (无需 NCU)

```python
"""快速测量单个 kernel 配置的延迟."""
import os, subprocess, sys

SHAPE = dict(num_expert=192, num_topk=8, hidden=4096, n_tp=384)

TEMPLATE = r"""
import sys
sys.path.insert(0, "/workspace/hpc_ref_copy/build/lib.linux-x86_64-cpython-312")
import hpc, torch
torch.manual_seed(0)
g, h, n_tp, topk, bs = {g}, {h}, {n_tp}, {topk}, {bs}
dtype = torch.float8_e4m3fn
topk_ids = torch.randint(0, g, (bs, topk), dtype=torch.int32, device='cuda')
topk_ids, _ = torch.sort(topk_ids, 1)
x_fp8 = (torch.randn((bs, h), dtype=torch.bfloat16, device='cuda')/100).to(dtype)
x_scale = torch.randn((bs, h//128), dtype=torch.float, device='cuda')
gw = (torch.randn((g, n_tp*2, h), dtype=torch.float, device='cuda')).to(dtype)
gws = torch.randn((g, n_tp*2//128, (h//128+3)//4*4), dtype=torch.float, device='cuda')
dw = (torch.randn((g, h, n_tp), dtype=torch.float, device='cuda')).to(dtype)
dws = torch.randn((g, h//128, (n_tp//128+3)//4*4), dtype=torch.float, device='cuda')
topk_s = torch.randn((bs, topk), dtype=torch.float, device='cuda')/topk

fn = lambda: torch.ops.hpc.fuse_moe_blockwise_cp_async(
    x_fp8, x_scale, gw, gws, dw, dws, topk_ids, topk_s, None, 0, g, None)
for _ in range(50): fn()
torch.cuda.synchronize()
N = 400
ev_s = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
ev_e = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
for i in range(N):
    ev_s[i].record(); fn(); ev_e[i].record()
torch.cuda.synchronize()
t = sorted(ev_s[i].elapsed_time(ev_e[i])*1000 for i in range(N))
print(f'RESULT bs={{bs}} med={{t[N//2]:.2f}} p25={{t[N//4]:.2f}} '
      f'p75={{t[3*N//4]:.2f}} min={{t[0]:.2f}}')
"""

for bs in [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
    src = TEMPLATE.format(
        g=SHAPE["num_expert"], h=SHAPE["hidden"],
        n_tp=SHAPE["n_tp"], topk=SHAPE["num_topk"], bs=bs)
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    out = subprocess.run([sys.executable, "-c", src],
                         env=env, capture_output=True, text=True, timeout=300)
    for line in out.stdout.splitlines():
        if line.startswith("RESULT"):
            print(line)
            break
    else:
        print(f"BS={bs} FAILED: {out.stderr[-200:]}")
```
