# data_loader.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from constants import BATCH_SIZE, CIFAR_MEAN, CIFAR_STD

def get_cifar_data_transforms():
    # create a set of augmentations we want to do to our training and data set.
    # for the training set we randomly flip the image to better generalize
    # and also randomly crop out a small portion of it
    # then we normalize the rgb values to be ~N(0,1)
    # the values for normalization can be derived from the dataset itself
    # or found here https://stackoverflow.com/questions/66678052/how-to-calculate-the-mean-and-the-std-of-cifar10-data
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=CIFAR_MEAN,
            std=CIFAR_STD
        ),
    ])

    # We do not augment the data if testing our classifier because we do not want to hinder the model in any way
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=CIFAR_MEAN,
            std=CIFAR_STD
        ),
    ])

    return transform_train, transform_test


def get_cifar_data_sets(transform_train, transform_test):
    # pytorch has some datasets built into it that make it extremely easy to download an use
    # this is the CIFAR10 Datasets which contains images for 10 different classes that we can then classify
    # we can decide whether to download the training or testing set and keep them seperate
    # we can also choose to download the datasets to disk to make loading them faster in the future
    # finally we can apply the transforms we created earlier to the data as it comes in
    train_set = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform_train,
    )
    test_set = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform_test,
    )

    return train_set, test_set


def cifar10_loaders(batch_size=BATCH_SIZE):
    transform_train, transform_test = get_cifar_data_transforms()
    train_set, test_set = get_cifar_data_sets(transform_train, transform_test)

    # dataloaders make it easy to iterate over all of the training data
    # batch size specifies how many data samples to load for each call of the loader
    # shuffle makes sure we get the samples in a different random order for each epoch
    # num workers creates sub processes to help load the data faster
    # that is useful because if we are training on a gpu, we can set aside a number of threads dedicated to transfering data over
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size,
        shuffle=True, 
        pin_memory=True,
        num_workers=4
    
    )
    test_loader = DataLoader(
        test_set, 
        batch_size=batch_size,
        shuffle=False, 
        pin_memory=True,
        num_workers=4
    )

    return train_loader, test_loader


def cifar10_loaders_ddp(world_size, rank, batch_size=BATCH_SIZE):
    transform_train, transform_test = get_cifar_data_transforms()

    if rank == 0:
        train_set, test_set = get_cifar_data_sets(transform_train, transform_test)

    # Wait for rank 0 to finish downloading
    if world_size > 1:
        dist.barrier()

    # Now all other ranks can safely load the downloaded data
    if rank != 0:
        train_set, test_set = get_cifar_data_sets(transform_train, transform_test)
    
    # Create DistributedSampler for training data
    # This divides the dataset among processes and handles shuffling
    # shuffle=True means data is shuffled before dividing among processes
    train_sampler = DistributedSampler(
        train_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    # Create DistributedSampler for test data
    # shuffle=False for test data to ensure consistent evaluation
    test_sampler = DistributedSampler(
        test_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Create DataLoaders with DistributedSampler
    # When using DistributedSampler, set shuffle=False in DataLoader
    # The sampler handles shuffling instead
    # pin_memory=True speeds up data transfer to GPU
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=train_sampler,  # Use sampler instead of shuffle
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        sampler=test_sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader


def cifar10_loaders_pipeline(world_size, rank, batch_size=BATCH_SIZE, shuffle_seed=42):
    # Middle ranks don't need data loaders
    if rank != 0 and rank != world_size - 1:
        return None, None
    
    transform_train, transform_test = get_cifar_data_transforms()
    train_set, test_set = get_cifar_data_sets(transform_train, transform_test)
    
    # Create generator with fixed seed for reproducible shuffling
    generator = torch.Generator()
    generator.manual_seed(shuffle_seed)
    
    # Create DataLoaders with RandomSampler using the same seed
    # This ensures rank 0 and last rank get the same batches in the same order
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=False,
        sampler=RandomSampler(train_set, generator=generator),
        pin_memory=True,
        num_workers=4,
        drop_last=True  # Drop last incomplete batch for consistent pipeline stages
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        drop_last=True
    )
    
    return train_loader, test_loader

