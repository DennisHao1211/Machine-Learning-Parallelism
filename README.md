# ML Parallelism Workshop - Setup & Usage Guide

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Training Methods & GPU Allocation](#training-methods--gpu-allocation)
3. [Model Weight Files](#model-weight-files)
4. [Training Results](#training-results)

---

## Environment Setup

### Important Note

Store conda environments and caches in `scratch`, not in `home`, to avoid exceeding the home directory quota.

### Step 1: Create directories in scratch

```bash
mkdir -p /home/hice1/hhao40/scratch/conda_envs
mkdir -p /home/hice1/hhao40/scratch/conda_pkgs
mkdir -p /home/hice1/hhao40/scratch/pip_cache

conda config --add envs_dirs /home/hice1/hhao40/scratch/conda_envs
conda config --add pkgs_dirs /home/hice1/hhao40/scratch/conda_pkgs
export PIP_CACHE_DIR=/home/hice1/hhao40/scratch/pip_cache
```

### Step 2: Load Anaconda

Required every time you open a new session (PACE module system requirement):

```bash
module load anaconda3
```

### Step 3: Create conda environment

Only needs to be done once:

```bash
conda create -p /home/hice1/hhao40/scratch/conda_envs/ml_workshop python=3.9 -y
```

### Step 4: Activate environment

Required every time you open a new session:

```bash
conda activate /home/hice1/hhao40/scratch/conda_envs/ml_workshop
```

### Step 5: Install dependencies

Only needs to be done once:

```bash
pip install -r /home/hice1/hhao40/Workshop-MLParallelism/requirements.txt
```

Installs: `torch`, `torchvision`, `numpy`, `matplotlib`

### Step 6: Verify installation

```bash
python -c "import torch; import torchvision; import numpy; import matplotlib; print('OK')"
```

If the output is `OK`, the environment is ready.

---

## Training Methods & GPU Allocation

Before every training run, execute the following:

```bash
module load anaconda3
conda activate /home/hice1/hhao40/scratch/conda_envs/ml_workshop
cd /home/hice1/hhao40/Workshop-MLParallelism/old/src
```

### Method 1: Single GPU Training

Allocate resources:

```bash
salloc --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=4 --mem=32G --time=3:00:00
```

Start training:

```bash
python train.py
```

- Prints Training Accuracy, Test Accuracy, and Epoch time after each epoch
- Saves weights to `cifar_cnn.pth` when complete

### Method 2: Data Parallelism (DDP)

Allocate resources:

```bash
salloc --gres=gpu:4 --ntasks-per-node=4 --cpus-per-task=4 --mem=32G --time=3:00:00
```

Start training:

```bash
python -m torch.distributed.run --nproc_per_node=4 train_ddp.py
```

- Each of the 4 GPUs holds a full copy of the model and processes a different subset of data
- Effectively processes 4x more data per step compared to single GPU
- Saves weights to `cifar_cnn.pth` when complete — this will overwrite single GPU weights, so back up first if needed:

```bash
cp cifar_cnn.pth cifar_cnn_singlegpu.pth
```

### Method 3: Pipeline Parallelism

Allocate resources:

```bash
salloc --gres=gpu:3 --ntasks-per-node=3 --cpus-per-task=4 --mem=32G --time=3:00:00
```

Start training:

```bash
python -m torch.distributed.run --nproc_per_node=3 train_pipeline.py
```

- The model is split across 3 GPUs by layer (Stage 1 / 2 / 3)
- Only prints Training Accuracy — Test Accuracy is not implemented (see Training Results for explanation)
- Each GPU saves its own stage weights separately when complete

---

## Model Weight Files

### Pre-trained Weights

Pre-trained weights from a completed run of all three training methods are available in `pretrained_weights/`. You can use these directly for inference without training from scratch.

| File | Training Method | Description |
|------|----------------|-------------|
| `cifar_cnn_singlegpu.pth` | Single GPU | Weights trained on 1 GPU — Test Accuracy: 91.32% |
| `cifar_cnn_ddp.pth` | DDP 4 GPU | Weights trained with data parallelism — Test Accuracy: 90.96% |
| `cifar_cnn.pth.stage0.pth` | Pipeline GPU 0 | Stage 1 weights (Conv1 + Conv2) |
| `cifar_cnn.pth.stage1.pth` | Pipeline GPU 1 | Stage 2 weights (Conv3 + Conv4) |
| `cifar_cnn.pth.stage2.pth` | Pipeline GPU 2 | Stage 3 weights (FC1 + FC2 + FC3) |

### Your Own Training Weights

When you run any of the training scripts, weights are saved to `old/src/cifar_cnn.pth` (single GPU / DDP) or `old/src/cifar_cnn.pth.stage*.pth` (pipeline).

### Loading Pipeline Weights for Inference

For inference, merge the three stage weights back into a single model:

```python
model = StagedCIFARNet()
model.stage1.load_state_dict(torch.load("cifar_cnn.pth.stage0.pth"))
model.stage2.load_state_dict(torch.load("cifar_cnn.pth.stage1.pth"))
model.stage3.load_state_dict(torch.load("cifar_cnn.pth.stage2.pth"))
```

---

## Running Inference

Before running inference, make sure the environment is activated:

```bash
module load anaconda3
conda activate /home/hice1/hhao40/scratch/conda_envs/ml_workshop
cd /home/hice1/hhao40/Workshop-MLParallelism/old/src
```

### Using Single GPU or DDP Weights

The inference script reads from `cifar_cnn.pth` in the current directory. Copy the desired pre-trained weight file there first, then run inference:

```bash
# Using single GPU weights
cp ../../pretrained_weights/cifar_cnn_singlegpu.pth cifar_cnn.pth
python inference.py

# Using DDP weights
cp ../../pretrained_weights/cifar_cnn_ddp.pth cifar_cnn.pth
python inference.py
```

### Using Pipeline Weights

```bash
cp ../../pretrained_weights/cifar_cnn.pth.stage0.pth .
cp ../../pretrained_weights/cifar_cnn.pth.stage1.pth .
cp ../../pretrained_weights/cifar_cnn.pth.stage2.pth .
python inference_pipeline.py
```

Both scripts print the predicted class and true label for a randomly selected test image, and save the result image to a `.png` file in the same directory.

---

## Training Results

| Method | Test Accuracy | Time per Epoch |
|--------|--------------|----------------|
| Single GPU | 91.32% | 2.80s |
| DDP 4 GPU | 90.96% | 0.99s |
| Pipeline 3 GPU | N/A | 2.28s |

> **Why is Pipeline Test Accuracy N/A?**
> During pipeline training, the model is split across 3 GPUs — each GPU only holds one stage and cannot run a full forward pass independently. Evaluating on the test set would require passing data sequentially through all 3 GPUs using `dist.send` / `dist.recv`, which requires an additional coordination layer. This was omitted in the current implementation of `train_pipeline.py`. Note that inference after training is still possible by merging all stage weights back onto a single device (see `inference_pipeline.py`).
