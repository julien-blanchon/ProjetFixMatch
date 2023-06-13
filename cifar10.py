import logging

# data and math
import math
import numpy as np

# image stuff
from PIL import Image

# dataset and transforms
from torchvision import datasets
from torchvision import transforms

# randaugment
from randaugment import RandAugmentMC

# typing
from typing import Callable, List, Tuple

# deep
import torch

# other
from utils import seedEverything



def compute_mean_std(trainLoader) -> Tuple[List[float], List[float]]:
    # initialize the list of means and stds
    mean, std = torch.zeros(3), torch.zeros(3)

    # iterate over the dataset and compute the sum of each channel
    for images, _ in trainLoader:
        mean+= torch.mean(images, dim=[0,2,3])
        std+= torch.std(images, dim=[0,2,3])
    
    # compute the mean and std
    mean = mean/len(trainLoader)
    std = std/len(trainLoader)

    return mean, std

# pre-computed values during the course of the project
# mean = [0.4913966, 0.48215377, 0.44651437]
# std = [0.246344, 0.24280126, 0.26067406]

def get_cifar10(root='./data'):
    # seed everything
    seedEverything()

    # base dataset with no transformations
    base_dataset = datasets.CIFAR10(root=root, train=True, download=True)

    # computing mean and std
    mean, std = compute_mean_std(base_dataset)

    # define transformations
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.125, 0.125)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    transform_validation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    

    # transform_unlabeled = transforms.Compose([
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.RandomAffine(degrees=10, translate=(0.125, 0.125)),
    #     RandAugmentMC(n=2, m=10),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=mean, std=std),
    # ])

