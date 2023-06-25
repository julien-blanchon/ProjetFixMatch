# system
import sys
sys.path.append('..')

# Contains Fixmatch classical algorithm
import torch

from utils.model import ConvNN

from tqdm.notebook import tqdm

from utils.utils import (
    seedEverything,
    weak_transform,
    strong_transform,
    normalize
)

from querying_fcts import (
    information_density,
    least_confidence
)

import numpy as np

from typing import Callable

def set_device() -> torch.device:
    # Set device
    if ((int(torch.__version__.split(".")[0]) >= 2) or (int(torch.__version__.split(".")[1]) >= 13)) and torch.has_mps:
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

# Create a new labeled dataset using active learning
def create_labeled_dataset_active_learning(dataset, selected_indices):
    dataset = torch.utils.data.Subset(dataset, selected_indices)
    return dataset

# MASK
def mask(
    model: ConvNN, 
    weak_unlabeled_data: torch.Tensor, 
    tau: float = 0.95) -> torch.Tensor:
    with torch.no_grad():
        model.train()

        qb = model(weak_unlabeled_data)

        qb = torch.softmax(qb, dim=1)

        max_qb, qb_hat = torch.max(qb, dim=1)

        idx = max_qb > tau
        qb_hat = qb_hat[idx]

    return qb_hat.detach(), idx, max_qb.detach()

def fixmatch_train_AL(
        model: ConvNN,
        labeled_dataloader: torch.utils.data.DataLoader,
        unlabeled_dataloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        labeled_criterion: torch.nn.Module,
        unlabeled_criterion: torch.nn.Module,
        query_function: Callable,
        target_prop: float,
        trainset_sup: torch.utils.data.Dataset,
        trainset_unsup: torch.utils.data.Dataset,
        trainset: torch.utils.data.Dataset,
        mean: float,
        std: float,
        lambda_u: int = 1,
        mu: int = 7,
        tau: float = 0.95,
        batch_size: int = 64,
        name: str = '') -> None:

    print("Start training")

    # setup for reproducibility
    device = set_device()
    
    current_prop = 0.005
    
    fulldata_length = len(unlabeled_dataloader.dataset)
    currentdata_length = int(fulldata_length * current_prop)
    data_added_per_epoch = 50

    epochs_init = 100
    epochs_final = 100
    epochs_AL = int((fulldata_length - currentdata_length) / data_added_per_epoch) * 10
    
    epochs = epochs_init + epochs_final + epochs_AL
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    
    seedEverything()

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    uncertainty = 0
    AL_activated = False

    step = 0
    for epoch in range(epochs):
        model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        running_n_unlabeled = 0
        running_accuracy = 0

        # initialize the progress bar
        pbar = tqdm(zip(labeled_dataloader, unlabeled_dataloader), total=min(len(labeled_dataloader), len(unlabeled_dataloader)), unit="batch", desc=f"Epoch {epoch: >5}")

        # loop through batch
        for i, (labeled_data, unlabeled_data) in enumerate(pbar):
            # Get labeled and unlabeled data
            labeled_inputs, labels = labeled_data[0].to(device), labeled_data[1].to(device)
            unlabeled_inputs, _ = unlabeled_data[0].to(device), unlabeled_data[1].to(device)
            step += len(labeled_inputs)
            
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Apply weak augmentation to labeled data
            weak_labeled_inputs = weak_transform(mean, std)(labeled_inputs)

            # Apply strong augmentation + weak augmentation to unlabeled data
            weak_unlabeled_inputs = weak_transform(mean, std)(unlabeled_inputs)
            strong_unlabeled_inputs = strong_transform(mean, std)(unlabeled_inputs)


            # Compute mask, confidence
            pseudo_labels, idx, max_qb = mask(model, weak_unlabeled_inputs, tau)
            strong_unlabeled_inputs = strong_unlabeled_inputs[idx]

            n_labeled, n_unlabeled = weak_labeled_inputs.size(0), strong_unlabeled_inputs.size(0)

            if n_unlabeled != 0:
                # Concatenate labeled and unlabeled data
                inputs_all = torch.cat((weak_labeled_inputs, strong_unlabeled_inputs))
                labels_all = torch.cat((labels, pseudo_labels))

                # forward pass
                outputs = model(inputs_all)

                # split labeled and unlabeled outputs
                labeled_outputs, unlabeled_outputs = outputs[:n_labeled], outputs[n_labeled:]

                # compute losses
                labeled_loss = torch.sum(labeled_criterion(labeled_outputs, labels)) / batch_size
                unlabeled_loss = torch.sum(unlabeled_criterion(unlabeled_outputs, pseudo_labels)) / (mu * batch_size)

                # compute total loss
                loss = labeled_loss + lambda_u * unlabeled_loss

                # compute accuracy
                total += labels_all.size(0)
                correct += (outputs.argmax(dim=1) == labels_all).sum().item()
            
            else:
                # forward pass
                labeled_outputs = model(weak_labeled_inputs)

                # compute loss
                labeled_loss = torch.sum(labeled_criterion(labeled_outputs, labels)) / batch_size
                unlabeled_loss = torch.tensor(0, device=device)

                # compute total loss
                loss = labeled_loss + lambda_u * unlabeled_loss

                # compute accuracy
                total += labels.size(0)
                correct += (labeled_outputs.argmax(dim=1) == labels).sum().item()
            
            # backward pass + optimize
            loss.backward()
            optimizer.step()

            # update statistics
            running_loss += loss.item()
            running_n_unlabeled += n_unlabeled
            running_accuracy += 100 * correct / total

            # update progress bar
            pbar.set_postfix({
                "total loss": loss.item(),
                "labeled loss": labeled_loss.item(),
                "unlabeled loss": unlabeled_loss.item(),
                "accuracy": 100 * correct / total,
                "confidence": torch.mean(max_qb).item(),
                "lab_prop": current_prop,
                "uncertainty": uncertainty,
                "n_unlabeled": running_n_unlabeled,
                "AL_activ": AL_activated,
                "step": step,
                "lr": optimizer.param_groups[0]['lr']
            })
            # scheduler step
            if scheduler is not None and (not AL_activated):
                scheduler.step()
                
        # AL step, start adding labels after 50 epochs
        if epoch >= epochs_init and current_prop < target_prop and epoch <= epochs_init + epochs_AL and epoch%10 == 0:
            AL_activated = True
            # compute querying
            selected_indices, uncertainty = query_function(model, unlabeled_dataloader, k_samp=data_added_per_epoch, mean=mean, std=std)
            
            # select indices from unlabeled dataset
            trainset_sup_new = create_labeled_dataset_active_learning(trainset_unsup, selected_indices)
            
            # concat new trainset with labeled trainset
            trainset_sup = torch.utils.data.ConcatDataset([trainset_sup, trainset_sup_new])

            # create labeled dataloader
            labeled_dataloader = torch.utils.data.DataLoader(trainset_sup, batch_size=batch_size, shuffle=True, num_workers=0)

            current_prop = len(trainset_sup) / len(trainset)
        else:
            AL_activated = False
            uncertainty = 0
        
        # update loss
        train_losses.append(running_loss / (i + 1))
        train_accuracies.append(running_accuracy / (i + 1))

        

        
        # Evaluate the model on the test set
        model.eval()  # Set the model to evaluation mode
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                # normalize
                images = normalize(mean, std)(images)
                
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
            
            test_accuracy = 100.0 * test_correct / test_total
            print(f'Test Accuracy: {test_accuracy}%')

            # update loss
            test_losses.append(torch.sum(labeled_criterion(outputs, labels)).item() / batch_size)
            test_accuracies.append(test_accuracy)

    print('Finished Training')

    # save model
    torch.save(model.state_dict(), f'./results/models/model_DA_{name}.pth')

    # save results
    torch.save(train_losses, f'./results/metrics/train_losses_DA_{name}.pth')
    torch.save(test_losses, f'./results/metrics/test_losses_DA_{name}.pth')
    torch.save(train_accuracies, f'./results/metrics/train_accuracies_DA_{name}.pth')
    torch.save(test_accuracies, f'./results/metrics/test_accuracies_DA_{name}.pth')