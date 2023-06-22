# system
import sys
sys.path.append('..')

# Contains Fixmatch classical algorithm
import torch

from utils.model import ConvNN

from tqdm.notebook import tqdm

from utils.utils import (
    seedEverything,
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

def supervised_train(
        model: ConvNN,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        criterion: torch.nn.Module,
        mean: float,
        std: float,
        epochs: int =300,
        name: str = '') -> None:
    
    print("Start training")

    # setup for reproducibility
    device = set_device()
    seedEverything()

    # losses
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        correct = 0
        total = 0

        # initialize progress bar 
        pbar = tqdm(trainloader, total=len(trainloader), unit="batch", desc=f"Epoch {epoch: >5}")

        # loop through batch
        for i, data in enumerate(pbar):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # normalize
            inputs = normalize(mean, std)(inputs)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Update statistics
            running_loss += loss.item()

            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            running_accuracy += 100 * correct / total

            current_lr = optimizer.param_groups[0]['lr']

            # Print progress
            pbar.set_postfix({
                "loss": loss.item(),
                "accuracy": 100 * correct / total,
                "lr": current_lr.item()
            })

            if scheduler is not None:
                scheduler.step()

        # update loss
        train_losses.append(running_loss / len(trainloader))
        train_accuracies.append(running_accuracy / (i+1))
        
        
        

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
            test_losses.append(criterion(outputs, labels).item())
            test_accuracies.append(test_accuracy)

        
    print('Finished Training')

    # save model
    torch.save(model.state_dict(), f'./results/models/model_{name}.pth')

    # save results
    torch.save(train_losses, f'./results/metrics/train_losses_{name}.pth')
    torch.save(test_losses, f'./results/metrics/test_losses_{name}.pth')
    torch.save(train_accuracies, f'./results/metrics/train_accuracies_{name}.pth')
    torch.save(test_accuracies, f'./results/metrics/test_accuracies_{name}.pth')