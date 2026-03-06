# train_pipeline
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe

import time
import os
import warnings

warnings.filterwarnings('ignore', category=FutureWarning, module='torch.distributed.pipelining')

from constants import LEARNING_RATE, MOMENTUM, WEIGHT_DECAY, EPOCHS, MODEL_WEIGHTS_PATH, BATCH_SIZE
from data_loader import cifar10_loaders_pipeline
from models import StagedCIFARNet


def setup():
    # same as ddp,
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", device_id=torch.device(f"cuda:{local_rank}"))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, local_rank, world_size


def cleanup():
    dist.destroy_process_group()


def manual_model_split(rank, num_stages, device):
    model = StagedCIFARNet()
    
    # each gpu gets one portion of our model
    if rank == 0:
        model_stage = model.stage1
    elif rank == 1:
        model_stage = model.stage2
    else:  # rank == 2
        model_stage = model.stage3
    
    # move only the portion of the model this gpu cares about to the gpu
    model_stage.to(device)
    
    # PipelineStage is where all of the magic happens
    # it is responsible for receiving and sending activations to the next/previous stages
    # and also computing gradients and sending them backwards
    pipeline_stage = PipelineStage(
        model_stage,
        rank,
        num_stages,
        device,
    )
    
    return pipeline_stage, model_stage


def main():
    rank, local_rank, world_size = setup()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"Pipeline Parallelism on {world_size} GPUs")

    train_loader, test_loader = cifar10_loaders_pipeline(world_size, rank, BATCH_SIZE)
    
    if train_loader is not None:
        num_batches = len(train_loader)
    else:
        num_batches = 0
    
    # Create pipeline stage for this rank
    pipeline_stage, model_stage = manual_model_split(rank, world_size, device)
    
    # Loss and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model_stage.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Broadcast number of batches in the training set so each rank knows how many it will process
    if rank == 0:
        num_batches_tensor = torch.tensor([num_batches], dtype=torch.long, device=device)
    else:
        num_batches_tensor = torch.tensor([0], dtype=torch.long, device=device)
    dist.broadcast(num_batches_tensor, src=0)
    num_batches = num_batches_tensor.item()
    
    # Number of microbatches
    # a micro batch is a split of a regular batch
    # so in this example we have 256 images in a batch so a micro batch would have 32 images
    # microbatches allow us to schedule and send smaller batches of activations to better use our gpu's
    n_microbatches = 8
    
    # schedule g pipe takes in our pipeline that we created and is then responsible for actually coordinating all of the data
    # responsible for actually splitting our batches into the microbatches
    # running the microbatches through one pipeline stage at the end
    # and the finally running all of the gradients through backwards
    # example schedule
    # GPU 0: F1   F2   F3   F4   idle idle idle idle B4   B3   B2   B1
    # GPU 1: idle F1   F2   F3   F4   idle idle B4   B3   B2   B1   idle
    # GPU 2: idle idle F1   F2   F3   F4   B4   B3   B2   B1   idle idle
    schedule = ScheduleGPipe(pipeline_stage, n_microbatches=n_microbatches, loss_fn=loss_function)

    for epoch in range(EPOCHS):
        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{EPOCHS}")
            start_time = time.time()

        # put the model in training mode again
        model_stage.train()
        
        # Track accuracy on last rank (where we have predictions)
        if rank == world_size - 1:
            train_correct = 0
            train_total = 0
        
        if rank == 0:
            train_iter = iter(train_loader)
            
            # this loop chucks all of our batches for an entire epoch into the first gpu
            for batch_idx in range(num_batches):
                optimizer.zero_grad()
                
                # first gpu, starts off the process, we only care about the images not the label
                images, _ = next(train_iter)
                images = images.to(device)
                
                # first gpu schedules with the images
                schedule.step(images)
                # adjust the weights
                optimizer.step()
                
        # last gpu
        elif rank == world_size - 1:
            train_iter = iter(train_loader)
            
            for batch_idx in range(num_batches):
                optimizer.zero_grad()

                # in the last gpu we care about the labels
                _, labels = next(train_iter)
                labels = labels.to(device)
                
                # last gpu computes the loss
                output = schedule.step(target=labels, losses=[])
                optimizer.step()
                
                # Track accuracy
                _, predicted = torch.max(output, 1)
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)
                
        else:
            for batch_idx in range(num_batches):
                optimizer.zero_grad()
                # any middle gpus just receive from the previous and send in their step
                schedule.step()
                optimizer.step()
            
        
        scheduler.step()
        
        # Print training accuracy from last rank
        if rank == world_size - 1:
            train_accuracy = 100.0 * train_correct / train_total
            print(f"Training Accuracy: {train_accuracy:.2f}%")

        if rank == 0:
            print(f"Epoch time: {(time.time() - start_time):.2f}s")

        # no test loop this time because we would have to make another scheduler etc, but its possible to do so

    # Each rank saves its stage
    stage_save_path = f"{MODEL_WEIGHTS_PATH}.stage{rank}.pth"
    torch.save(model_stage.state_dict(), stage_save_path)

    cleanup()


if __name__ == "__main__":
    main()