# system
import sys
sys.path.append('..')

# Contains Fixmatch classical algorithm
import torch
from torch.utils.data import DataLoader

from utils.model import ConvNN

from tqdm.notebook import tqdm

from utils.utils import (
    seedEverything,
    weak_transform,
    strong_transform,
    normalize
)

def set_device() -> torch.device:
    # Set device
    if ((int(torch.__version__.split(".")[0]) >= 2) or (int(torch.__version__.split(".")[1]) >= 13)) and torch.has_mps:
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

import numpy as np

from typing import Callable

def information_density(
    model: ConvNN,
    unlabeled_dataloader: DataLoader,
    k_samp: int,
    mean: float,
    std: float,
    K_transform: int = 5) -> torch.Tensor:
    
    device = set_device()
    model.eval()

    with torch.no_grad():
        for i, (inputs, _) in enumerate(unlabeled_dataloader):
            inputs = inputs.to(device)

            logits_mean = torch.zeros((len(inputs), 10)).to(device)

            for k in range(K_transform):
                # compute weak transform of inputs
                weak_transformed_inputs = weak_transform(mean, std)(inputs)

                # compute predictions of weak transformed inputs
                logits = model(weak_transformed_inputs)

                logits_mean += logits
            
            logits_mean /= K_transform

            qb = torch.softmax(logits_mean, dim=1).to(device)
        
        # get top 2 values of qb along dim=1
        qb1, qb2 = torch.topk(qb, k=2, dim=1, sorted=True)[0][:, 0], torch.topk(qb, k=2, dim=1, sorted=True)[0][:, 1]

        uncertainty = 1 - (qb1 - qb2)

        # get bottom k_samp indices
        uncert, idx = torch.topk(uncertainty, k=k_samp, dim=0, sorted=True, largest=False)
        
        # get top k_samp indices
        # uncert, idx = torch.topk(uncertainty, k=k_samp, dim=0, sorted=True)

        return idx, torch.mean(uncert).item()
    
def least_confidence(
    model: ConvNN,
    unlabeled_dataloader: torch.utils.data.DataLoader,
    k_samp: int,
    mean: float,
    std: float) -> torch.Tensor:
    
    probs = []
    device = set_device()

    model.eval()

    with torch.no_grad():
        for i, (inputs, _) in enumerate(unlabeled_dataloader):
            inputs = inputs.to(device)

            # inputs transforms
            weak_transformed_inputs = weak_transform(mean, std)(inputs)

            logits = model(weak_transformed_inputs)
            qb = torch.softmax(logits, dim=1)
            probs.append(qb.cpu().data)
    
    # transform to tensor
    probs_tensor = torch.cat(probs, dim=0)

    U = probs_tensor.max(1)[0]

    selected_indices = U.sort()[1][:k_samp]

    return selected_indices, torch.mean(torch.sort(U, descending=False)[0][:k_samp]).item()