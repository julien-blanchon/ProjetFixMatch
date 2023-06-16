# deep learning
# typing
from typing import Callable

# plotting
import matplotlib.pyplot as plt

# data
import numpy as np
import torch
from matplotlib.figure import Figure
from torch.utils.data import DataLoader, Dataset, random_split

# torchvision
from torchvision import datasets
from torchvision import transforms as T

# random and os
import random
import os

DEFAULT_RANDOM_SEED = 2021

def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
# torch random seed
import torch
def seedTorch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
      
# basic + tensorflow + torch 
def seedEverything(seed=DEFAULT_RANDOM_SEED):
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

def plot_transform(
    images: torch.Tensor, 
    transform: Callable[[torch.Tensor], torch.Tensor], 
    transform_name: str or None = None) -> Figure:    

    assert images.shape[0] >= 4, "Not enough images to plot"

    fig, axes = plt.subplots(2, 4, figsize=(10,5))
    images = images[:4]
    images_transformed = transform(images)
    transform_name = transform_name if transform_name is not None else ''

    for i in range(4):
        axes[0][i].imshow(images[i].cpu().numpy().transpose(1, 2, 0))
        axes[0][i].set_title(f"Original")
        axes[0][i].axis('off')

        axes[1][i].imshow(images_transformed[i].cpu().numpy().transpose(1, 2, 0))
        axes[1][i].set_title(f"Transformed {transform_name}")
        axes[1][i].axis('off')

    return fig