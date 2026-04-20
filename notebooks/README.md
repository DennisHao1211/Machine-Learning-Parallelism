# Running Notebooks on Kaggle

## Setup Guide

[Open the PDF guide](KaggleSetup.pdf)

## DDP vs Tensor Parallel vs Pipeline Parallel

| Dimension | DDP (Data Parallel) | Tensor Parallel (TP) | Pipeline Parallel (PP) |
|------|------|------|------|
| **Primary split unit** | Data batch is split across GPUs | Individual weight tensors inside a layer are split across GPUs | Model is split by layer groups (stages) across GPUs |
| **Model replica per GPU** | Full model on every GPU | Partial tensors for TP layers; non-TP layers can remain replicated | Only the local stage is stored on each GPU |
| **Input seen by each GPU in one step** | Different mini-batch shard per GPU | Same mini-batch on all participating GPUs in the TP group | Different micro-batches in flight through different stages |
| **Typical communication point** | Gradient all-reduce after backward | Collectives inside forward/backward of TP layers (all-gather/all-reduce) | Activation/gradient transfer between adjacent stages (`send`/`recv`) |
| **Memory effect** | Does not reduce parameter memory per GPU | Reduces parameter memory for sharded layers | Reduces parameter memory by stage partitioning |
| **Throughput scaling pattern** | Usually best first scaling step when model fits on one GPU | Scales large matrix ops when single-layer tensors are too large | Improves device utilization with micro-batch pipelining, but has bubble overhead |
| **Main bottleneck** | Gradient synchronization bandwidth | Intra-layer communication latency/bandwidth | Pipeline bubbles and stage load imbalance |
| **Implementation complexity** | Lowest | Medium | Highest |
| **Failure mode if configured incorrectly** | Wrong global batch / LR scaling, rank sync issues | Different input batches across TP ranks produce invalid partial results | Stage mismatch/deadlock on `send`/`recv`, poor stage balance |
| **Best use case** | Model fits on one GPU and you want faster training via more data parallel workers | Very large layers (e.g., large MLP/attention projections) do not fit efficiently on one GPU | Very deep/large models where splitting by stages is natural |
| **Notebook in this folder** | `02-data-parallel.ipynb` | `04-tensor-parallel.ipynb` | `03-pipeline-parallel.ipynb` |
