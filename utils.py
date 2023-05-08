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

# class Data():
#     def __init__(self, batch_size: int = 64, splits: list = [0.90, 0.10]) -> None:
#         self.batch_size = batch_size
#         self.trainset = datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=T.ToTensor())

#         self.trainset, self.validset = random_split(self.trainset, splits)

#         self.testset = datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=T.ToTensor())
                                        
#         self.trainloader = DataLoader(self.trainset, batch_size=self.batch_size,
#                                           shuffle=True, num_workers=4)

#         self.validloader = DataLoader(self.validset, batch_size=self.batch_size,
#                                           shuffle=True, num_workers=4)                                  
                                       
#         self.testloader = DataLoader(self.testset, batch_size=self.batch_size,
#                                          shuffle=True, num_workers=4)

#         self.classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#     def get_train_loader(self) -> DataLoader:
#         return self.trainloader

#     def get_valid_loader(self) -> DataLoader:
#         return self.validloader

#     def get_test_loader(self) -> DataLoader:
#         return self.testloader

#     def get_classes(self) -> tuple:
#         return self.classes
    
# class DataFixMatch(Data):
#     def __init__(self, data: Data, batch_size: int = 64, splits: list = [0.90, 0.10]) -> None:
#         self.batch_size = batch_size

#         self.trainset_sup, self.trainset_unsup = random_split(data.trainset, splits)

#         self.trainloader_sup = DataLoader(self.trainset_sup, batch_size=self.batch_size, shuffle=True, num_workers=4)
#         self.trainloader_unsup = DataLoader(self.trainset_unsup, batch_size=self.batch_size, shuffle=True, num_workers=4)

#     def get_trainsup_loader(self) -> DataLoader:
#         return self.trainloader_sup

#     def get_trainunsup_loader(self) -> DataLoader:
#         return self.trainloader_unsup

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
    transform_name: str | None = None) -> Figure:    

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