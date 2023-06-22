# deep learning
# typing
from typing import Callable, List, Tuple

# plotting
import matplotlib.pyplot as plt

# data
import numpy as np
import torch
from matplotlib.figure import Figure

# random and os
import random
import os

# import torchvision
from torchvision import transforms

# random augmentation
from utils.randaugment import RandAugmentMC

DEFAULT_RANDOM_SEED = 2021

def seedBasic(seed: int = DEFAULT_RANDOM_SEED) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
# torch random seed
import torch
def seedTorch(seed: int = DEFAULT_RANDOM_SEED) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
      
# basic + tensorflow + torch 
def seedEverything(seed: int = DEFAULT_RANDOM_SEED) -> None:
    seedBasic(seed)
    seedTorch(seed)


def plot_images(
    images: torch.Tensor, 
    true_labels: torch.Tensor, 
    predicted_labels: torch.Tensor,
    classes: tuple,
    figure_name: str) -> Figure:

    assert images.shape[0] >= 16, "Not enough images to plot"
    assert images.shape[0] == true_labels.shape[0] == predicted_labels.shape[0], "Number of images and labels do not match"

    fig, axes = plt.subplots(4, 4, figsize=(10,10))
    axes = axes.ravel()

    for i in range(16):
        axes[i].imshow(images[i].cpu().numpy().transpose(1, 2, 0))
        axes[i].set_title(f"True: {classes[true_labels[i]]}\nPredicted: {classes[predicted_labels[i]]}")
        axes[i].axis('off')
        fig.subplots_adjust(hspace=0.5)
    fig.suptitle(figure_name, fontsize=16)
    return fig

# compute mean and std
def compute_mean_std(trainLoader: torch.utils.data.DataLoader) -> Tuple[List[float], List[float]]:
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

# data transform
def data_transform() -> Callable:
    return transforms.Compose([
        transforms.ToTensor(),
    ])

def normalize(mean: float, std: float) -> Callable:
    return transforms.Normalize(mean, std)

# define weak transform
def weak_transform(mean: float, std: float) -> Callable:
    
    weak_transform_function = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.125, 0.125)),
        transforms.Normalize(mean, std),
    ])

    return weak_transform_function

# define strong transform
def strong_transform(mean: float, std: float) -> Callable:
    strong_transform_function = transforms.Compose([
        RandAugmentMC(n=2, m=10),
        transforms.Normalize(mean, std)
    ])

    return strong_transform_function