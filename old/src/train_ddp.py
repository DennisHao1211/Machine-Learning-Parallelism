# train_ddp.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import time
import os

from constants import LEARNING_RATE, MOMENTUM, WEIGHT_DECAY, EPOCHS, MODEL_WEIGHTS_PATH
from data_loader import cifar10_loaders_ddp
from models import CIFARNet


def setup_ddp():
    # get what gpu this process is currently running with
    # local rank env variable is set by the torch runner
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # tells the process that the default gpu is local_rank i.e 0,1,2 etc instead of just 0
    # so every process is using a different gpu
    torch.cuda.set_device(local_rank)

    # Initialize the process group
    # NCCL backend is optimized for NVIDIA GPUs and is the fastest for multi-GPU training
    # init method tells us where to look to initialize - we want to look in the environment variables
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        device_id=torch.device(f"cuda:{local_rank}")
    )

    # Get rank information from environment variables
    # if we have just 1 node, then rank and local rank should be the same
    rank = dist.get_rank()  # Global process ID (0 to world_size-1)
    world_size = dist.get_world_size()  # Total number of processes
    
    return rank, local_rank, world_size


def cleanup_ddp():
    dist.destroy_process_group()


def main():
    # Setup distributed training
    rank, local_rank, world_size = setup_ddp()

    # Normally when device is "cuda" this actually gets translated to "cuda:0"
    # which just means use the first GPU. With our multiple GPU setup:
    device = torch.device(f"cuda:{local_rank}")

    # Only print from rank 0 to avoid cluttered output
    if rank == 0:
        print(f"Training with DDP on {world_size} GPUs")

    # Print the device on every process to verify mapping
    print(f"Using device: {device}")

    train_loader, test_loader = cifar10_loaders_ddp(world_size, rank)

    # Create model and move it to the correct GPU
    model = CIFARNet().to(device)

    # Wrap model with DDP
    # This handles gradient synchronization across processes
    # device ids is what gpu this ddp model must explicitly deal with and run through 
    # we must specify it so each gpu gets its own model
    model = DDP(model, device_ids=[local_rank])

    # Loss function and optimizer same as before
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    # Learning rate scheduler same as before
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    # Training, only gets accuract on one gpu however
    # in theory you could reduce the accuracies from all gpu's and average them
    for epoch in range(EPOCHS):
        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{EPOCHS}")
            start_time = time.time()

        # Set epoch for DistributedSampler to ensure different shuffling
        # without it on every epoch every gpu would have the same random sample
        train_loader.sampler.set_epoch(epoch)

        # Train
        model.train()
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = loss_function(outputs, labels)

            # DDP automatically synchronizes gradients during backward()
            loss.backward()
            optimizer.step()

            # Track training accuracy
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        train_accuracy = 100.0 * train_correct / train_total

        # Update learning rate
        scheduler.step()

        if rank == 0:
            print(f"Training Accuracy: {train_accuracy:.2f}%")

        # Validate
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                test_correct += (predicted == labels).sum().item()
                test_total += labels.size(0)

        test_accuracy = 100.0 * test_correct / test_total

        if rank == 0:
            print(f"Test Accuracy: {test_accuracy:.2f}%")
            print(f"Epoch time: {(time.time() - start_time):.2f}s")

    # Save model (only from rank 0 to avoid conflicts)
    if rank == 0:
        torch.save(model.module.state_dict(), MODEL_WEIGHTS_PATH)
        print(f"\nSaved model weights to {MODEL_WEIGHTS_PATH}")

    # Clean up
    cleanup_ddp()

if __name__ == "__main__":
    main()