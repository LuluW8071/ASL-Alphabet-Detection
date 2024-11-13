import torch
import numpy as np
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from torchmetrics import Accuracy


device = "cuda" if torch.cuda.is_available() else "cpu"
torchmetrics_accuracy = Accuracy(task = "multiclass", num_classes = 36).to(device)

def train_step(model, dataloader, loss_fn, optimizer, device, experiment, epoch):
    """
    Perform a single training step.

    Args:
        model (torch.nn.Module): The neural network model to train.
        dataloader (torch.utils.data.DataLoader): The training data loader.
        loss_fn (torch.nn.Module): The loss function to use for training.
        optimizer (torch.optim): The optimizer to use for training.
        device (torch.device): The device to use for training.

    Returns:
        Tuple[float, float]: A tuple containing the average training loss and accuracy.

    """
    # Put model on train mode
    model.train()

    # Setup train loss and accuracy values
    train_loss, train_acc = 0, 0
    train_losses, train_accuracies = [], []

    # Loop through dataloader data batches
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)   # Send data to target device

        y_pred = model(X)                   # Forward pass
        loss = loss_fn(y_pred, y)           # Calculate loss
        train_loss += loss                  # Accumulate loss
        optimizer.zero_grad()               # Optimizer zero grad
        loss.backward()                     # Loss backward
        optimizer.step()                    # Optimizer step

        # Calculate and ammulate accuracy metrics across all batches 
        y_pred_class = torch.argmax(torch.softmax(y_pred,
                                                dim=1), dim=1)
        train_acc += torchmetrics_accuracy(y_pred_class, y)
    
    # Adjust metrics to get avg. loss and accuracy per batch 
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    metrics_dict = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
    }

    # Log metrics dictionary to Comet ML
    experiment.log_metrics(metrics_dict, step=epoch)

    train_losses.append(train_loss.item())
    train_accuracies.append(train_acc.item())

    return train_losses, train_accuracies


def test_step(model, dataloader, loss_fn, device, experiment, epoch):
    """
    Perform a single testing step.

    Args:
        model (torch.nn.Module): The neural network model to test.
        dataloader (torch.utils.data.DataLoader): The testing data loader.
        loss_fn (torch.nn.Module): The loss function to use for testing.
        device (torch.device): The device to use for testing.

    Returns:
        Tuple[float, float]: A tuple containing the average testing loss and accuracy.

    """
    # Put model on evaluation mode
    model.eval()

    # Setup test loss and accuracy values
    test_loss, test_acc = 0, 0
    test_losses, test_accuracies = [], []
    preds_list, labels_list = [], []

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through dataloader data batches
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)       # Send data to target device
            # print(y, y.shape)
            test_pred_logits = model(X)             # Forward pass
            loss = loss_fn(test_pred_logits, y)     # Calculate loss
            test_loss += loss                       # Accumulate loss

            # Calculate and ammulate accuracy metrics across all batches 
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += torchmetrics_accuracy(test_pred_labels, y)

            # Collect predictions and labels for confusion matrix
            preds_list.extend(test_pred_labels.cpu().numpy())
            labels_list.extend(y.cpu().numpy())

        # Adjust metrics to get avg. loss and accuracy per batch 
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)

        # Log metrics to a dictionary
        metrics_dict = {
            "test_loss": test_loss,
            "test_accuracy": test_acc,
        }

        # Log metrics dictionary to Comet ML
        experiment.log_metrics(metrics_dict, step=epoch)

        test_losses.append(test_loss.item())
        test_accuracies.append(test_acc.item())

        return test_losses, test_accuracies, preds_list, labels_list
    


def train(model, train_dataloader, test_dataloader, classes,
          loss_fn, optimizer, epochs, device, experiment):
    """
    Train a neural network model for a given number of epochs.

    Args:
        model (torch.nn.Module): The neural network model to train.
        train_dataloader (torch.utils.data.DataLoader): The training data loader.
        test_dataloader (torch.utils.data.DataLoader): The testing data loader.
        loss_fn (torch.nn.Module): The loss function to use for training and testing.
        optimizer (torch.optim): The optimizer to use for training.
        epochs (int): The number of epochs to train for.
        device (torch.device): The device to use for training and testing.

    Returns:
        Dict[str, List[float]]: A dictionary containing the training and testing loss and accuracy for each epoch.

    """
    test_acc_min = 0

    # Create empty results dictionary 
    results = {"train_losses_history": [],
               "train_accuracies_history": [],
               "test_losses_history": [],
               "test_accuracies_history": []
    }

    # Loop through train_steps and test_steps for no of epochs
    for epoch in tqdm(range(epochs)):
        train_losses, train_accuracies = train_step(model,
                                           train_dataloader,
                                           loss_fn,
                                           optimizer,
                                           device,
                                           experiment,
                                           epoch)
        test_losses, test_accuracies, preds_list, labels_list = test_step(model,
                                        test_dataloader,
                                        loss_fn,
                                        device,
                                        experiment,
                                        epoch)

         # Calculate avg. loss and accuracy
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_train_accuracy = sum(train_accuracies) / len(train_accuracies)
        avg_test_loss = sum(test_losses) / len(test_losses)
        avg_test_accuracy = sum(test_accuracies) / len(test_accuracies)

        print(f'\nTrain loss: {avg_train_loss:.4f} ---- Train acc: {avg_train_accuracy:.4f}')
        print(f'Test loss: {avg_test_loss:.4f} ---- Test acc: {avg_test_accuracy:.4f}\n')

        # Save model if avg_test_accuracy is higher
        if avg_test_accuracy > test_acc_min:
            torch.save(model.state_dict(), 'best_model.pth')
            test_acc_min = avg_test_accuracy
            print(f"Saved best model at epoch: {epoch}\n")

            # Log the confusion matrix to Comet
            experiment.log_confusion_matrix(
                y_true=np.array(labels_list),
                y_predicted=np.array(preds_list),
                labels=classes,
            )

        results["train_losses_history"].append(avg_train_loss)
        results["train_accuracies_history"].append(avg_train_accuracy)
        results["test_losses_history"].append(avg_test_loss)
        results["test_accuracies_history"].append(avg_test_accuracy)

    return results