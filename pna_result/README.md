# PNA Training: Performance, Energy, and Efficiency Analysis

This report presents a profiling study of the PNA (Principal Neighbourhood Aggregation) graph neural network trained on the ZINC molecular dataset. We examine how batch size affects training latency, GPU utilisation, and energy consumption, and investigate the sources of execution-time variability observed during training.

All experiments were run on a single SLURM-managed GPU node. Four batch sizes are studied: **512**, **1024**, **2048**, and **4096** (the largest that fits in GPU memory). Batch size 4096 is the primary focus of this report.

---

## 1. Step-Level Timing at Batch Size 4096

At batch size 4096, the ZINC training set (10 000 graphs) yields **3 steps per epoch**: two full batches of 4096 graphs and one partial batch of 1808 graphs. This creates a distinctive repeating pattern in the step-time trace.

### The "Spike" Pattern Is Batch-Shape, Not a Bug

The total-time plot shows a striking alternation between ~775 ms steps and ~285 ms steps. At first glance this looks like a performance anomaly, but overlaying batch-shape metadata reveals the cause immediately: the short steps are simply the **partial final batch** of each epoch, which contains fewer than half the graphs (1808 vs 4096) and proportionally fewer nodes and edges.

![Total Step Execution Time + Batch Shape](simple/plots/bs4096/pna_simple_bs4096_wk0_total_time_batch_shape.png)

The dashed lines on the secondary axis show that every third step (the short bar) has roughly half the nodes (~25 000 vs ~57 000) and half the edges (~52 000 vs ~118 000). Since GNN computation scales with the number of nodes and edges in the batch, the step time scales proportionally. There is no true "spike" here — the training loop is behaving exactly as expected.

### Time Breakdown

The backward pass dominates computation at this batch size, accounting for **61.3%** of mean step time (375.6 ms), while the forward pass takes **37.1%** (227.6 ms). The optimizer step is negligible at 0.7%.

![Time Breakdown Donut](simple/plots/bs4096/pna_simple_bs4096_wk0_pancake.png)

---

## 2. Step-Level Timing at Batch Size 512

At batch size 512, each epoch has 20 steps and the execution time is much more uniform — except for **periodic spikes** that nearly triple the step latency, from the ~110 ms baseline to 240–290 ms.

Unlike the batch-size-driven variation at bs4096, these spikes are caused by **Python's garbage collector**. The spike experiment confirms this.

### Identifying GC as the Cause

We ran the same workload with GC event logging enabled. Annotating gen-2 collection events onto the time-breakdown plot reveals a perfect correlation: every spike aligns with a gen-2 GC sweep.

![GC On — Annotated (bs512)](spike/plots/pna_spike_bs512_wk0_breakdown_gc_on_annotated.png)

Each gen-2 collection pauses the process for 130–180 ms, which shows up as a sudden expansion of the optimizer/other region in the stacked breakdown. These collections occur roughly every 20 steps.

### Disabling GC Eliminates the Spikes

Running the same workload with automatic garbage collection disabled produces a perfectly flat execution trace:

![GC Off (bs512)](spike/plots/pna_spike_bs512_wk0_breakdown_gc_off.png)

The step time holds steady at ~108 ms with no variation, confirming that GC is the sole source of the observed spikes at this batch size.

---

## 3. Batch-Size Comparison

### Epoch Latency

Despite much longer per-step times, larger batch sizes complete each epoch **faster** because they process the same dataset in fewer steps.

![Epoch Latency vs Batch Size](plots/bs_epoch_latency.png)

| Batch Size | Mean Step Time | Steps/Epoch | Mean Epoch Latency |
|:----------:|:--------------:|:-----------:|:------------------:|
| 512        | 113 ms         | 20          | **2.3 s**          |
| 1024       | 148 ms         | 10          | **1.5 s**          |
| 2048       | 322 ms         | 5           | **1.6 s**          |
| 4096       | 613 ms         | 3           | **1.8 s**          |

Batch size 1024 achieves the lowest epoch latency. At 2048 and 4096, per-step compute grows faster than the reduction in step count, so epoch latency rises slightly — but all three larger sizes remain well below bs512.

### GPU Utilisation

GPU utilisation shows a non-monotonic pattern: it **drops** from bs512 to bs1024, then **climbs steeply** to 60% at bs4096.

![GPU Utilisation vs Batch Size](plots/bs_gpu_util.png)

| Batch Size | GPU Util |
|:----------:|:--------:|
| 512        | 26.6%    |
| 1024       | 13.5%    |
| 2048       | 16.2%    |
| 4096       | 60.0%    |

**Why the initial drop?** At bs512, each step is short (~110 ms) and there are 20 steps per epoch. The utilisation sampler (500 ms interval) often captures the GPU mid-computation, giving a moderate reading. At bs1024, per-step time grows to ~150 ms but is still short enough that the GPU frequently sits idle between CUDA kernel launches during data loading and Python overhead; samples are more likely to land in these gaps.

**Why the jump at 4096?** At this batch size the individual CUDA kernels (scatter, gather, matrix multiplications over ~57 000 nodes and ~118 000 edges) are large enough to keep the GPU's streaming multiprocessors continuously occupied. The computation becomes **compute-bound** rather than **launch-overhead-bound**, crossing a threshold where GPU occupancy stays high throughout each step.

### Energy per Epoch

Larger batch sizes are more **energy-efficient** per epoch, consuming up to 33% less energy:

![Energy per Epoch vs Batch Size](plots/bs_energy_total.png)

| Batch Size | Energy/Epoch |
|:----------:|:------------:|
| 512        | 0.450 mWh    |
| 1024       | 0.379 mWh    |
| 2048       | 0.385 mWh    |
| 4096       | 0.300 mWh    |

This follows directly from higher GPU utilisation: a batch of 4096 graphs keeps the hardware busy doing useful work rather than idling at near-constant base power draw. Even though each step consumes more energy in absolute terms, fewer steps per epoch and better hardware utilisation more than compensate.

### Energy Breakdown by Hardware

At bs4096, the **CPU dominates energy consumption** at 66.1%, with the GPU accounting for 32.0% and RAM a negligible 1.8%.

![Energy Breakdown (bs4096)](carbon/plots/pna_carbon_bs4096_wk0_pancake_energy_hardware.png)

This is characteristic of graph neural network workloads where significant CPU time is spent on irregular data preparation (graph batching via PyTorch Geometric's `Batch.from_data_list`) while the GPU kernels themselves, though intensive, are relatively short-lived.

---

## 4. Measurement Overhead

Instrumenting training with utilisation sampling and energy tracking adds minimal overhead. At bs4096, mean step latency across four configurations is:

![Measurement Overhead](plots/overhead.png)

| Configuration    | Mean Step Latency | Overhead vs Baseline |
|:----------------:|:-----------------:|:--------------------:|
| Simple (timers only) | 612.9 ms      | —                    |
| Baseline (gc-manual) | 610.8 ms      | reference            |
| Util measurement     | 611.8 ms      | +0.2%                |
| Carbon (CodeCarbon)  | 615.5 ms      | +0.8%                |

The heaviest instrumentation (CodeCarbon energy tracking) adds less than 1% overhead, confirming that the measurement tooling does not meaningfully distort the results presented in this report.

---

## 5. Summary of Findings

1. **Batch size 4096** achieves the highest GPU utilisation (60%) and lowest energy per epoch (0.300 mWh), making it the most hardware-efficient configuration — despite not having the fastest epoch latency.

2. **Execution-time spikes have different root causes** depending on batch size:
   - At **bs4096**, the periodic short steps are simply **partial batches** at epoch boundaries (1808 vs 4096 graphs). This is expected and not a performance problem.
   - At **bs512**, spikes are caused by **Python gen-2 garbage collection** pauses of 130–180 ms. Disabling GC eliminates them entirely.

3. **GPU utilisation follows a non-monotonic curve** from bs512 to bs4096. Small and medium batch sizes (512–2048) underutilise the GPU because CUDA kernels are too small to saturate the hardware. Only at bs4096 does the per-step workload cross the threshold for sustained GPU occupancy.

4. **Larger batches are more energy-efficient** per epoch because they amortise fixed idle power draw over more useful computation — a direct consequence of higher GPU utilisation.

5. **Measurement overhead is negligible** (<1%), so all results can be taken at face value.
