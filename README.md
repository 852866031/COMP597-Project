# PNA GNN: Performance, Energy, and Efficiency Analysis

This report presents a profiling study of the **Principal Neighbourhood Aggregation (PNA)** graph neural network trained on a molecular property prediction task. We examine how batch size affects training latency, GPU utilisation, and energy consumption, and investigate the sources of step-time variability at different operating points.

All experiments run on a single SLURM-managed GPU node with **2 DataLoader workers** to overlap data preparation with GPU computation. Four batch sizes are studied: **512**, **1024**, **2048**, and **4096** (the largest that fits in GPU memory). Batch size **4096** is the primary focus; **512** is the secondary reference for contrast.

---

## Table of Contents

1. [The PNA Workload](#1-the-pna-workload)
2. [Training at Batch Size 4096](#2-training-at-batch-size-4096)
3. [Training at Batch Size 512 — GC Spike Attribution](#3-training-at-batch-size-512--gc-spike-attribution)
4. [Batch-Size Comparison](#4-batch-size-comparison)
5. [Energy and Carbon](#5-energy-and-carbon)
6. [Measurement Overhead](#6-measurement-overhead)
7. [Summary](#7-summary)

---

## 1. The PNA Workload

### What Is PNA?

**Principal Neighbourhood Aggregation (PNA)** is a GNN architecture that applies *multiple* aggregation functions (mean, min, max, std) simultaneously at each layer and scales their outputs by node degree. This multi-aggregator design makes it more expressive than standard message-passing networks.

The model is trained on a molecular dataset where each graph represents a molecule (nodes = atoms, edges = bonds). The regression target is a quantum-chemical property. The model uses **64 PNA convolution layers**, **4 aggregators**, and **3 scalers**, totalling ~4 million parameters.

### Training Pipeline

Each training step follows three phases:

```
 FORWARD       64 layers of neighbourhood aggregation + global pooling + L1 loss
 BACKWARD      autograd through 64 conv layers over irregular sparse graphs
 OPTIMIZER     Adam weight update + cosine-annealing LR schedule
```

Data loading is handled by PyTorch Geometric's `DataLoader` with **2 worker processes** that pre-fetch and collate batches in parallel with GPU computation. This pipelining is critical for GPU utilisation — without it, the main thread alternates between CPU-bound collation and GPU dispatch, leaving the GPU idle during data preparation.

---

## 2. Training at Batch Size 4096

At batch size 4096, the training dataset yields **3 steps per epoch**: two full batches of 4096 graphs and one partial batch of 1808 graphs (the remainder). This creates a distinctive repeating pattern.

### The Step-Time Pattern Is Driven by Batch Shape

The total-time plot shows a striking alternation between ~775 ms steps and ~285 ms steps:

![Total Step Execution Time (bs4096)](pna_result/simple/plots/bs4096/pna_simple_bs4096_wk2_total_time.png)

Overlaying the batch-shape metadata immediately reveals the cause — the short steps are simply the **partial final batch** of each epoch:

![Total Step Execution Time + Batch Shape](pna_result/simple/plots/bs4096/pna_simple_bs4096_wk2_total_time_batch_shape.png)

Every third step has roughly half the nodes (~25 000 vs ~57 000) and half the edges (~52 000 vs ~118 000). Since GNN computation scales with the number of nodes and edges, step time scales proportionally. There is no "spike" — the training loop is behaving exactly as expected given the dataset size.

### Time Breakdown

| | |
|:---:|:---:|
| ![Breakdown (bs4096)](pna_result/simple/plots/bs4096/pna_simple_bs4096_wk2_breakdown.png) | ![Loss (bs4096)](pna_result/simple/plots/bs4096/pna_simple_bs4096_wk2_loss.png) |
| *Execution time breakdown per step* | *Training loss per step* |

<table>
<tr>
<td width="340"><img src="pna_result/simple/plots/bs4096/pna_simple_bs4096_wk2_pancake.png" width="330"></td>
<td valign="top">

The backward pass dominates at **61.2%** of mean step time (375.5 ms), while the forward pass takes **37.2%** (228.2 ms). The optimizer step is negligible at **1.0%** (3.9 ms).

This split is characteristic of deep GNNs: backpropagating gradients through 64 convolution layers over irregular sparse graphs is significantly more expensive than the forward pass. At large batch sizes the backward pass must manage large intermediate tensors across ~57 000 nodes, making gradient accumulation the dominant cost.

</td>
</tr>
</table>

### Hardware Utilisation at Batch Size 4096

| | |
|:---:|:---:|
| ![GPU util (bs4096)](pna_result/utils/plots/pna_utils_bs4096_wk2_steps_util_gpu.png) | ![CPU util (bs4096)](pna_result/utils/plots/pna_utils_bs4096_wk2_steps_util_cpu.png) |
| *GPU utilisation — avg 92.3%* | *Per-process CPU utilisation — avg 81.9%* |

![RAM util (bs4096)](pna_result/utils/plots/pna_utils_bs4096_wk2_steps_util_ram.png)

GPU utilisation averages **92.3%** at this batch size — remarkably high for a GNN workload. With 2 DataLoader workers pre-fetching batches, the GPU rarely stalls waiting for data. Each full batch contains ~57 000 nodes and ~118 000 edges, producing CUDA kernels large enough to keep the streaming multiprocessors continuously saturated.

---

## 3. Training at Batch Size 512 — GC Spike Attribution

At batch size 512, each epoch has 20 steps and execution time is far more uniform — except for **periodic spikes** that significantly exceed the ~105 ms baseline:

| | |
|:---:|:---:|
| ![Total Time (bs512)](pna_result/simple/plots/bs512/pna_simple_bs512_wk2_total_time.png) | ![Breakdown (bs512)](pna_result/simple/plots/bs512/pna_simple_bs512_wk2_breakdown.png) |
| *Step execution time — note the spike* | *Time breakdown — spike appears across all substeps* |

Unlike the batch-shape-driven variation at bs4096, these spikes are caused by **Python's garbage collector**.

### Identifying GC as the Cause

We ran the same workload with GC event logging enabled. Annotating generation-2 collection events onto the time-breakdown reveals the correlation — **spikes align with gen-2 GC sweeps**:

![GC On — Annotated (bs512)](pna_result/spike/plots/pna_spike_bs512_wk2_breakdown_gc_on_annotated.png)

Gen-2 collections pause the process for **130–180 ms**. These pauses are triggered by PyTorch Geometric's `DataLoader`, which creates many short-lived Python objects (graph tensors, batch index tensors) every step. These objects get promoted through Python's generational GC until they trigger a gen-2 sweep.

### Disabling GC Eliminates the Spikes

Running with automatic GC disabled produces a **perfectly flat** execution trace:

![GC Off (bs512)](pna_result/spike/plots/pna_spike_bs512_wk2_breakdown_gc_off.png)

Step time holds steady at ~105 ms with no variation, confirming that GC is the **sole source** of the spikes at this batch size.

<table>
<tr>
<td width="340"><img src="pna_result/simple/plots/bs512/pna_simple_bs512_wk2_pancake.png" width="330"></td>
<td valign="top">

At bs512, the time split differs from bs4096: the forward pass takes **55.3%** (58.3 ms) and the backward pass **38.5%** (40.6 ms). This inversion occurs because small batches produce small graphs where the forward aggregation overhead (64 layers of scatter/gather over many small neighbourhoods) dominates relative to the backward pass, which has less intermediate state to accumulate gradients over.

As batch size grows, backward scales super-linearly due to larger intermediate tensors and more complex autograd graph traversal, eventually overtaking the forward pass.

</td>
</tr>
</table>

### The Same GC Pattern Appears at Batch Size 4096

GC events also affect bs4096 training when GC is enabled. The spike experiment at bs4096 confirms this and shows that disabling GC produces an equally flat trace:

| | |
|:---:|:---:|
| ![GC On (bs4096)](pna_result/spike/plots/pna_spike_bs4096_wk2_breakdown_gc_on.png) | ![GC Off (bs4096)](pna_result/spike/plots/pna_spike_bs4096_wk2_breakdown_gc_off.png) |
| *GC on: variability from gen-2 collections* | *GC off: flat ~775 ms trace* |

---

## 4. Batch-Size Comparison

### Epoch Latency

Despite much longer per-step times, larger batch sizes complete each epoch **faster** because they process the same dataset in fewer steps:

![Epoch Latency vs Batch Size](pna_result/plots/bs_epoch_latency.png)

| Batch Size | Mean Step Time | Steps/Epoch | Epoch Latency |
|:----------:|:--------------:|:-----------:|:-------------:|
| 512        | 105 ms         | 20          | **2.1 s**     |
| 1024       | 148 ms         | 10          | **1.5 s**     |
| 2048       | 322 ms         | 5           | **1.6 s**     |
| 4096       | 614 ms         | 3           | **1.8 s**     |

Batch size 1024 achieves the lowest epoch latency (1.5 s). Beyond that, per-step compute grows faster than the step-count reduction, so epoch latency rises slightly — but all larger sizes remain well below bs512.

### GPU Utilisation Scales with Batch Size

With 2 DataLoader workers, GPU utilisation **increases monotonically** with batch size — a clean scaling relationship:

| | |
|:---:|:---:|
| ![GPU Util vs BS](pna_result/plots/bs_gpu_util.png) | ![CPU Util vs BS](pna_result/plots/bs_cpu_util.png) |
| *GPU utilisation scales smoothly from 45% to 92%* | *CPU utilisation remains high across all sizes (81–91%)* |

| Batch Size | GPU Util | CPU Util |
|:----------:|:--------:|:--------:|
| 512        | 44.6%    | 90.8%    |
| 1024       | 55.9%    | 84.0%    |
| 2048       | 63.2%    | 81.4%    |
| 4096       | 92.3%    | 81.9%    |

**Why GPU utilisation increases**: larger batches produce larger batched graphs with more nodes and edges. This creates larger CUDA kernels with higher arithmetic intensity per launch, improving occupancy on the GPU's streaming multiprocessors. At bs4096 (~57 000 nodes, ~118 000 edges per batch), the kernels are large enough to keep the GPU near-continuously busy.

**The role of DataLoader workers**: with `num_workers=2`, data preparation is pipelined with GPU computation. This eliminates the idle gaps that plagued the single-threaded (`num_workers=0`) configuration. The clean monotonic scaling is a direct result of this pipelining — without it, GPU utilisation showed a non-monotonic dip at intermediate batch sizes where the GPU waited for the CPU to finish collating each batch.

**CPU utilisation**: remains high across all batch sizes (81–91%). The main process is consistently loaded by CUDA kernel dispatch, while the 2 worker processes handle data collation in parallel.

### Batch Sizes 1024 and 2048

| | |
|:---:|:---:|
| ![Total Time (bs1024)](pna_result/simple/plots/bs1024/pna_simple_bs1024_wk2_total_time.png) | ![Breakdown (bs1024)](pna_result/simple/plots/bs1024/pna_simple_bs1024_wk2_breakdown.png) |
| *bs1024: ~148 ms steps* | *bs1024: forward/backward transition zone* |

| | |
|:---:|:---:|
| ![Total Time (bs2048)](pna_result/simple/plots/bs2048/pna_simple_bs2048_wk2_total_time.png) | ![Breakdown (bs2048)](pna_result/simple/plots/bs2048/pna_simple_bs2048_wk2_breakdown.png) |
| *bs2048: ~322 ms steps, partial-batch dips visible* | *bs2048: backward pass starts dominating* |

These intermediate sizes show a transition: at bs1024 the forward and backward passes are roughly balanced, while at bs2048 the backward pass starts to dominate — reaching the 61/37 split seen at bs4096.

---

## 5. Energy and Carbon

### Energy per Epoch

Energy per epoch varies with batch size, with **bs1024 being the most energy-efficient** configuration:

![Energy per Epoch vs Batch Size](pna_result/plots/bs_energy_total.png)

| Batch Size | Energy/Epoch |
|:----------:|:------------:|
| 512        | 0.314 mWh    |
| 1024       | 0.264 mWh    |
| 2048       | 0.280 mWh    |
| 4096       | 0.310 mWh    |

The sweet spot is at bs1024: it has the fastest epoch latency (1.5 s) and the lowest energy consumption. At bs4096, although GPU utilisation is highest (92.3%), the longer per-step computation outweighs the step-count reduction, pushing total energy back up. The result is a U-shaped energy curve — an important finding for practitioners choosing batch sizes for energy-efficient training.

### Energy Breakdown by Hardware

| | |
|:---:|:---:|
| ![Energy Hardware (bs4096)](pna_result/carbon/plots/pna_carbon_bs4096_wk2_energy_hardware.png) | ![Energy Hardware (bs512)](pna_result/carbon/plots/pna_carbon_bs512_wk2_energy_hardware.png) |
| *bs4096: per-step energy by hardware* | *bs512: per-step energy by hardware* |

<table>
<tr>
<td width="340"><img src="pna_result/carbon/plots/pna_carbon_bs4096_wk2_pancake_energy_hardware.png" width="330"></td>
<td valign="top">

At bs4096, the **CPU dominates energy consumption** at **65.6%** (0.068 mWh per step), with the GPU accounting for **32.6%** (0.034 mWh) and RAM a negligible **1.8%**.

Despite the GPU running at 92.3% utilisation, CPU energy still dominates because CodeCarbon attributes CPU power at the constant TDP rate — the full rated thermal design power is charged for every second of wall-clock time. Since each bs4096 step takes ~614 ms, the CPU's constant power draw accumulates substantially.

The practical implication: **reducing wall-clock training time** is the primary lever for reducing both energy consumption and carbon emissions.

</td>
</tr>
</table>

### Energy per Hardware Component Across Batch Sizes

![Energy Hardware vs BS](pna_result/plots/bs_energy_hardware.png)

Per-step energy for all hardware components grows with batch size (larger batches = more computation per step), but CPU consistently dominates. The GPU's share increases at larger batch sizes as higher utilisation translates to higher power draw, narrowing the gap with the constant-TDP CPU attribution.

### Per-Step Energy Traces

| | |
|:---:|:---:|
| ![Energy Total (bs4096)](pna_result/carbon/plots/pna_carbon_bs4096_wk2_energy_total.png) | ![Carbon Total (bs4096)](pna_result/carbon/plots/pna_carbon_bs4096_wk2_carbon_total.png) |
| *bs4096: per-step energy (mWh)* | *bs4096: per-step CO₂ emissions (µg CO₂eq)* |

| | |
|:---:|:---:|
| ![Energy Total (bs512)](pna_result/carbon/plots/pna_carbon_bs512_wk2_energy_total.png) | ![Carbon Total (bs512)](pna_result/carbon/plots/pna_carbon_bs512_wk2_carbon_total.png) |
| *bs512: per-step energy (mWh)* | *bs512: per-step CO₂ emissions (µg CO₂eq)* |

<table>
<tr>
<td width="340"><img src="pna_result/carbon/plots/pna_carbon_bs512_wk2_pancake_energy_hardware.png" width="330"></td>
<td valign="top">

At bs512, the CPU's energy dominance is even more pronounced. The GPU operates at 44.6% utilisation — drawing proportionally less power — while the CPU's constant TDP attribution runs at full rate for every millisecond of wall-clock time.

The carbon emissions traces mirror the energy traces exactly (Quebec's grid carbon intensity is a constant factor), so the same conclusions apply: **choosing the right batch size matters for emissions**, and bs1024 is the sweet spot in this configuration.

</td>
</tr>
</table>

---

## 6. Measurement Overhead

Instrumenting training with utilisation sampling and energy tracking adds minimal overhead. At bs4096, mean step latency across configurations:

![Measurement Overhead](pna_result/plots/overhead.png)

| Configuration          | Mean Step Latency | Overhead |
|:----------------------:|:-----------------:|:--------:|
| Simple (timers only)   | 614.0 ms          | baseline |
| Util measurement       | 614.8 ms          | +0.1%    |
| Carbon (CodeCarbon)    | 615.9 ms          | +0.3%    |

All instrumentation adds less than **0.3% overhead**, confirming that the measurement tooling does not meaningfully distort any of the results presented in this report.

---

## 7. Summary

1. **Batch size 4096** achieves the highest GPU utilisation (**92.3%**) thanks to large CUDA kernels that saturate the GPU. With 2 DataLoader workers, data pipelining keeps the GPU near-continuously busy.

2. **Batch size 1024** is the energy sweet spot — lowest epoch latency (**1.5 s**) and lowest energy per epoch (**0.264 mWh**). At bs4096, high GPU utilisation doesn't fully compensate for longer steps.

3. **Step-time spikes have different root causes at different batch sizes:**
   - At **bs4096**, the periodic short steps are simply **partial batches** at epoch boundaries (1808 vs 4096 graphs).
   - At **bs512**, spikes are caused by **Python gen-2 garbage collection** pauses of 130–180 ms. Disabling GC eliminates them entirely.

4. **GPU utilisation scales monotonically** with batch size (44.6% → 55.9% → 63.2% → 92.3%) when DataLoader workers are used. This clean scaling is enabled by pipelining data preparation with GPU computation.

5. **CPU energy dominates** at all batch sizes (~66% at bs4096) due to CodeCarbon's constant-TDP attribution model. Reducing wall-clock training time remains the primary lever for cutting carbon emissions.

6. **Measurement overhead is negligible** (<0.3%), so all results can be taken at face value.
