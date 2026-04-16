# ML Parallelism Mini Workshop

## Objective

By the end of this workshop, you will:
- Train a neural network on PACE ICE supercomputer
- Save and use trained model weights locally
- Speed up training using data parallelism
- Split large models across multple gpus using pipeline parallelism

---

## Problem Overview

We'll work with the CIFAR-10 dataset which contains 60,000 32×32 color images across 10 classes:
- airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck


---

## Network Architecture

### CIFARNet Structure
Our CNN has:
- **4 Convolutional Layers** (with batch normalization and max pooling)
  - Conv1: 3 → 64 channels
  - Conv2: 64 → 128 channels
  - Conv3: 128 → 256 channels
  - Conv4: 256 → 512 channels
- **3 Fully Connected Layers** with dropout
  - FC1: 32,768 → 512
  - FC2: 512 → 256
  - FC3: 256 → 10 (output classes)

**Techniques Used**:
- Batch Normalization: Stabilizes training
- Dropout (50%): Prevents overfitting
- Data Augmentation: Random flips and crops
- Max Pooling: Reduces spatial dimensions

---

## PyTorch Basics

### Model Definition
Models extend `nn.Module` and implement:
- `__init__()`: Define layers
- `forward()`: Define data flow through network

```python
class CIFARNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        # ... more layers
    
    def forward(self, x):
        x = self.conv1(x)
        # ... forward pass
        return x
```

### Training Loop
```python
model.train()  # Enable dropout, batch norm updates
for images, labels in train_loader:
    optimizer.zero_grad()  # Clear gradients
    outputs = model(images)  # Forward pass
    loss = loss_function(outputs, labels)  # Compute loss
    loss.backward()  # Compute gradients
    optimizer.step()  # Update weights
```

### Saving/Loading Weights
```python
# Save
torch.save(model.state_dict(), 'model.pth')

# Load
model.load_state_dict(torch.load('model.pth'))
```

---

## Running the model locally
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train.py
```

--

## Getting Started with ICE

### Connecting to PACE
```bash
# Connect via SSH (requires VPN or eduroam)
ssh YOUR_USERNAME@login-ice.pace.gatech.edu
```

### Checking Available Resources
```bash
# View nodes, CPUs, memory, and GPUs
sinfo -o "%20N %10c %10m %25f %10G"

# Check GPU info (when in an interactive session)
nvidia-smi
```

---

## Transferring Files

### Option 1: Using rsync
```bash
# Transfer project to PACE
rsync -avz --progress --exclude-from='./Workshop1-IntroMl/.rsyncignore' \
  Workshop1-IntroML/ \
  YOUR_USERNAME@login-ice.pace.gatech.edu:~/Workshop1-IntroML/

# Download trained weights from PACE
rsync -avz --progress \
  YOUR_USERNAME@login-ice.pace.gatech.edu:~/Workshop1-IntroML/src/cifar_cnn.pth \
  ./Workshop1-IntroML/src/

# Download pipeline weights (all stages)
rsync -avz --progress \
  "YOUR_USERNAME@login-ice.pace.gatech.edu:~/Workshop1-IntroML/src/cifar_cnn.pth.stage*.pth" \
  ./Workshop1-IntroML/src/
```

**rsync flags**:
- `-a`: Archive mode (preserves timestamps, permissions)
- `-v`: Verbose output
- `-z`: Compress during transfer
- `--progress`: Show transfer progress

### Option 2: Using OnDemand
Navigate to [https://ondemand-ice.pace.gatech.edu](https://ondemand-ice.pace.gatech.edu)
- Use the file browser to upload/download files
- Provides a web interface for file management, and opening up a terminal through your web browser

### Option 3: Git
```bash
# On PACE
git clone YOUR_REPO_URL
```

---

## Running Jobs on PACE

### Interactive Sessions
For testing and debugging:

```bash
# --gres=gpu:<gpu-type>:<gpu-count>

# Single GPU session
salloc --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=4 --mem=32G --time=3:00:00

# Multiple GPU session (for DDP)
salloc --gres=gpu:4 --ntasks-per-node=4 --cpus-per-task=4 --mem=32G --time=3:00:00

# 3 GPU session (for pipeline parallelism)
salloc --gres=gpu:3 --ntasks-per-node=3 --cpus-per-task=4 --mem=32G --time=3:00:00
```

**Flag explanations**:
- `--gres=gpu:N`: Request N GPUs
- `--ntasks-per-node=N`: Number of processes per node
- `--cpus-per-task=N`: CPUs allocated per process
- `--mem=32G`: Memory allocation
- `--time=3:00:00`: Maximum time (HH:MM:SS)

<!-- ### Batch Jobs (SBATCH)
For long-running jobs, create a batch script:

```bash
#!/bin/bash
#SBATCH -J train_model          # Job name
#SBATCH --account=YOUR_ACCOUNT  # Account
#SBATCH -N 1                    # Number of nodes
#SBATCH --ntasks-per-node=1     # Tasks per node
#SBATCH --cpus-per-task=4       # CPUs per task
#SBATCH --mem=32G               # Memory
#SBATCH --gres=gpu:1            # GPUs
#SBATCH -t 3:00:00              # Time limit
#SBATCH -o train_%j.out         # Output file
#SBATCH -e train_%j.err         # Error file

module load anaconda3
conda activate ml_workshop

cd ~/Workshop1-IntroML/src
python train.py
```

Submit with:
```bash
sbatch job_script.sh
```

### Monitoring Jobs
```bash
# Check job status
squeue -u YOUR_USERNAME

# Cancel a job
scancel JOB_ID

# View job output
tail -f train_JOBID.out
```

For more details: [PACE Job Monitoring Guide](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042096) -->

---

## Setting Up Your Environment

### View Available Modules
```bash
module avail  # List all available software modules
```

### Load Python
```bash
# Load Anaconda - all relevant python interpreters + pip
module load anaconda3

# Install dependencies
pip install -r requirements.txt
```
---

## Training Your Model

### Single GPU Training
```bash
# In interactive session or batch job
python train.py
```

**What happens**:
1. Loads CIFAR-10 dataset with augmentations
2. Creates CIFARNet model
3. Trains for 100 epochs with SGD optimizer
4. Saves weights to `cifar_cnn.pth`

---

## Inference

### Running Inference on PACE
```bash
python inference.py
```

### Running Inference Locally
1. Download weights from PACE:
```bash
rsync -avz --progress \
  YOUR_USERNAME@login-ice.pace.gatech.edu:~/Workshop1-IntroML/src/cifar_cnn.pth \
  ./src/
```

2. Run locally:
```bash
# This will actually display the image on your computer which can't happen on pace
python inference.py
```

---

## Data Parallelism (DDP)

### What is Data Parallelism?
- **Copies** the entire model to each GPU
- Each GPU processes a different subset of data
- Gradients are synchronized across GPUs
- Effective for training large batches faster

### How DDP Works
1. Each GPU gets its own copy of the model
2. `DistributedSampler` divides dataset among GPUs
3. Forward pass computed independently
4. Gradients synchronized via `all-reduce`
5. All GPUs update weights identically

### Key DDP Components

#### Setup
```python
def setup_ddp():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return dist.get_rank(), local_rank, dist.get_world_size()
```

#### Wrap Model
```python
model = CIFARNet().to(device)
model = DDP(model, device_ids=[local_rank])
```

#### Use DistributedSampler
```python
train_sampler = DistributedSampler(
    train_set,
    num_replicas=world_size,
    rank=rank,
    shuffle=True
)

train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    sampler=train_sampler,
    shuffle=False  # Sampler handles shuffling
)

# Set epoch for proper shuffling
train_sampler.set_epoch(epoch)
```

### Running DDP

#### Interactive Session
```bash
# Request 4 GPUs
salloc --gres=gpu:4 --ntasks-per-node=4 --cpus-per-task=4 --mem=32G --time=3:00:00

# Run with torch distributed launcher
python -m torch.distributed.run --nproc_per_node=4 train_ddp.py
```

---

## Model Parallelism (Pipeline)

### What is Model Parallelism?
- **Splits** the model across multiple GPUs
- Each GPU holds a different part (stage) of the model
- Data flows through stages sequentially
- Useful when model is too large for one GPU

### Pipeline Parallelism
- Divides batches into **microbatches**
- Overlaps computation across stages
- Reduces GPU idle time

**Example schedule with 4 microbatches**:
```
    # GPU 0: F1   F2   F3   F4   idle idle idle idle B4   B3   B2   B1
    # GPU 1: idle F1   F2   F3   F4   idle idle B4   B3   B2   B1   idle
    # GPU 2: idle idle F1   F2   F3   F4   B4   B3   B2   B1   idle idle
```

### Model Splitting
The `StagedCIFARNet` is split into 3 stages:

**Stage 1 (GPU 0)**: First conv layers
```python
Conv2d(3→64) → Pool → BatchNorm → ReLU
Conv2d(64→128) → Pool → BatchNorm → ReLU
```

**Stage 2 (GPU 1)**: Later conv layers
```python
Conv2d(128→256) → BatchNorm → ReLU
Conv2d(256→512) → BatchNorm → ReLU → Flatten
```

**Stage 3 (GPU 2)**: Fully connected layers
```python
Linear(32768→512) → ReLU → Dropout
Linear(512→256) → ReLU → Dropout
Linear(256→10)
```

### Pipeline Components

#### PipelineStage
```python
pipeline_stage = PipelineStage(
    model_stage,  # The portion of model on this GPU
    rank,         # GPU ID
    num_stages,   # Total number of stages
    device
)
```

#### ScheduleGPipe
```python
schedule = ScheduleGPipe(
    pipeline_stage,
    n_microbatches=8,  # Split each batch into 8 microbatches
    loss_fn=loss_function
)
```

#### Training Logic
```python
if rank == 0:  # First GPU
    schedule.step(images)
elif rank == world_size - 1:  # Last GPU
    output = schedule.step(target=labels, losses=[])
else:  # Middle GPUs
    schedule.step()
```

### Running Pipeline Parallelism

```bash
# Request 3 GPUs
salloc --gres=gpu:3 --ntasks-per-node=3 --cpus-per-task=4 --mem=32G --time=3:00:00

# Load environment
module load anaconda3

# Run with 3 processes (one per stage)
python -m torch.distributed.run --nproc_per_node=3 train_pipeline.py
```

### Saving/Loading Pipeline Models

**After training**, each GPU saves its stage:
```
cifar_cnn.pth.stage0.pth  # GPU 0's stage
cifar_cnn.pth.stage1.pth  # GPU 1's stage
cifar_cnn.pth.stage2.pth  # GPU 2's stage
```

**For inference**:
```python
model = StagedCIFARNet()
model.stage1.load_state_dict(torch.load("cifar_cnn.pth.stage0.pth"))
model.stage2.load_state_dict(torch.load("cifar_cnn.pth.stage1.pth"))
model.stage3.load_state_dict(torch.load("cifar_cnn.pth.stage2.pth"))
```

Run inference:
```bash
python inference_pipeline.py
```

---

## Additional Resources

- [PACE ICE Main Documentation](https://gatech.service-now.com/technology?id=kb_article_view&sysparm_article=KB0042102)
- [PACE Resource Guide](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042095)
- [Job Monitoring Guide](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042096)
- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [PyTorch Pipeline Parallelism](https://pytorch.org/docs/stable/distributed.pipelining.html)

---