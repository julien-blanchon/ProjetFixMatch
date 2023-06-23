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

import numpy as np

def set_device() -> torch.device:
    # Set device
    if ((int(torch.__version__.split(".")[0]) >= 2) or (int(torch.__version__.split(".")[1]) >= 13)) and torch.has_mps:
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

def fixmatch_DA_train(
        model: ConvNN,
        labeled_dataloader: torch.utils.data.DataLoader,
        unlabeled_dataloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        labeled_criterion: torch.nn.Module,
        unlabeled_criterion: torch.nn.Module,
        mean: float,
        std: float,
        lambda_u: int = 1,
        mu: int = 7,
        tau: float = 0.95,
        batch_size: int = 64,
        epochs: int =300,
        name: str = '') -> None:
    
    CIFAR10_class_distribution = 0.10

    print("Start training")

    # setup for reproducibility
    device = set_device()
    seedEverything()

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        running_n_unlabeled = 0
        running_accuracy = 0
        moving_avg_pred_labeled = 0
        moving_avg_pred_unlabeled = 0

        # initialize the progress bar
        pbar = tqdm(zip(labeled_dataloader, unlabeled_dataloader), total=min(len(labeled_dataloader), len(unlabeled_dataloader)), unit="batch", desc=f"Epoch {epoch: >5}")

        # loop through batch
        for i, (labeled_data, unlabeled_data) in enumerate(pbar):
            # Get labeled and unlabeled data
            labeled_inputs, labels = labeled_data[0].to(device), labeled_data[1].to(device)
            unlabeled_inputs, _ = unlabeled_data[0].to(device), unlabeled_data[1].to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()

            # normalize labeled and unlabeled inputs
            _, unlabeled_inputs_norm = normalize(mean, std)(labeled_inputs), normalize(mean, std)(unlabeled_inputs)

            unlabeled_outputs_norm = model(unlabeled_inputs_norm)

            # Compute moving average of labeled and unlabeled predictions
            moving_avg_pred_unlabeled = (i * moving_avg_pred_unlabeled + unlabeled_outputs_norm.shape[0]) / (i + 1)

            # ratio
            ratio = CIFAR10_class_distribution / moving_avg_pred_unlabeled

            # Apply weak augmentation to labeled data
            weak_labeled_inputs = weak_transform(mean, std)(labeled_inputs)

            # Apply strong augmentation + weak augmentation to unlabeled data
            weak_unlabeled_inputs = weak_transform(mean, std)(unlabeled_inputs)
            strong_unlabeled_inputs = strong_transform(mean, std)(unlabeled_inputs)

            # prediction on weak augmented unlabeled data
            qb = model(weak_unlabeled_inputs)
            qb = torch.softmax(qb, dim=1)
            qb_norm = qb * ratio

            # normalize
            qb_tilde = qb_norm / torch.sum(qb_norm, dim=1, keepdim=True)

            # compute mask
            max_qb_tilde, qb_tilde_hat = torch.max(qb_tilde, dim=1)
            idx = max_qb_tilde > tau

            # pseudo labels
            pseudo_labels = qb_tilde_hat[idx]

            # mask strong augmented unlabeled data
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
                "confidence": torch.mean(max_qb_tilde).item(),
                "n_unlabeled": running_n_unlabeled,
                "lr": optimizer.param_groups[0]['lr']
            })
            # scheduler step
            if scheduler is not None:
                scheduler.step()
        
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