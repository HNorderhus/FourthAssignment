import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from src.models import ResNet20, ResNet8



def train_one_epoch(model, data_loader, criterion, optimizer, device):
    """
    Trains a given model for one epoch using the provided data loader, criterion, and optimizer.
    Args:
        model (nn.Module): The model to be trained.
        data_loader (DataLoader): The data loader providing the training data.
        criterion (nn.Module): The loss function to be used during training.
        optimizer (torch.optim.Optimizer): The optimizer to be used for updating the model's parameters.
        device (torch.device): The device on which the model is running (e.g., 'cpu' or 'cuda').
    Returns:
        float: The average loss per batch for the entire epoch.
    """
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(data_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(data_loader)
    return epoch_loss


def evaluate_one_epoch(model, data_loader, criterion, device):
    """
    Tests a given model for one epoch using the provided data loader and criterion, and computes the top-1 and top-5 accuracies.

    Args:
        model (nn.Module): The model to be tested.
        data_loader (DataLoader): The data loader providing the testing data.
        criterion (nn.Module): The loss function to be used during testing.
        device (torch.device): The device on which the model is running (e.g., 'cpu' or 'cuda').

    Returns:
        float: The average loss per batch for the entire epoch.
        float: The top-1 accuracy of the model on the test data.
        float: The top-5 accuracy of the model on the test data.
    """
    model.eval()
    running_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, top1_predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            top1_correct += (top1_predicted == labels).sum().item()

            # Compute top-5 accuracy
            _, top5_predicted = torch.topk(outputs.data, 5, dim=1)
            top5_correct += (labels.view(-1, 1) == top5_predicted).sum().item()

    epoch_loss = running_loss / len(data_loader)
    top1_accuracy = top1_correct / total
    top5_accuracy = top5_correct / total
    return epoch_loss, top1_accuracy, top5_accuracy


def train_and_evaluate_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, scheduler=None):
    """
    Trains a given model for a specified number of epochs using the provided data loader, criterion,
    and optimizer, and tracks the loss for each epoch.

    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): The data loader providing the training data.
        test_loader (DataLoader): The data loader providing the testing data.
        criterion (nn.Module): The loss function to be used during training and testing.
        optimizer (torch.optim.Optimizer): The optimizer to be used for updating the model's parameters.
        num_epochs (int): The number of epochs to train the model.
        device (torch.device): The device on which the model is running (e.g., 'cpu' or 'cuda').
        scheduler (torch.optim.lr_scheduler, optional): The learning rate scheduler (default is None).

    Returns:
        list: A list of the average loss per batch for each epoch.
        list: A list of the average loss per batch for each testing epoch.
        list: A list of the top-1 accuracy for each testing epoch.
        list: A list of the top-5 accuracy for each testing epoch.
    """
    train_losses = []
    test_losses = []
    test_top1_accuracies = []
    test_top5_accuracies = []
    model.to(device)

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_top1_accuracy, test_top5_accuracy = evaluate_one_epoch(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_top1_accuracies.append(test_top1_accuracy)
        test_top5_accuracies.append(test_top5_accuracy)

        if scheduler:
            scheduler.step(test_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, "
              f"Test Top-1 Accuracy: {test_top1_accuracy:.4f}, Test Top-5 Accuracy: {test_top5_accuracy:.4f}")

    return train_losses, test_losses, test_top1_accuracies, test_top5_accuracies


# The code below is to work task 5, to test whether the architecture works correctly

#if __name__ == "__main__":
#    # Device
#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    print(f"Device: {device}")

    # Transformations
#    resize = 112

#    test_transform = transforms.Compose([
#        transforms.Resize((resize, resize)),
#        transforms.ToTensor(),
#        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#    ])

#    train_transform = transforms.Compose([
#        transforms.RandomHorizontalFlip(0.5),
#        transforms.RandomRotation(15),
#        transforms.RandomResizedCrop(resize),
#        transforms.ToTensor(),
#        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#    ])

    # Dataset & Dataloader
#    num_classes = 10
#    trainset = CIFAR10(root="../data", train=True, download=True, transform=train_transform)
#    testset = CIFAR10(root="../data", train=False, download=True, transform=test_transform)

#    train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
#    test_loader = DataLoader(testset, batch_size=64, shuffle=False)

    # Model
#    model = ResNet8(3, num_classes=num_classes)

#    num_epochs = 1
#    criterion = torch.nn.CrossEntropyLoss()
#    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 28], gamma=0.1)

#    train_loss, test_loss, test_top1_acc, test_top5_acc = train_and_evaluate_model(model, train_loader, test_loader,
 #                                                                                  criterion, optimizer, num_epochs,
 #                                                                                  device)

    # Save training data
#    model_data = {
#        "name": "resnet20",
#        "train_losses": train_loss,
#        "test_losses": test_loss,
#        "test_accuracies": test_top1_acc,
 #       "test_top5_accuracies": test_top5_acc
 #   }

#    save_training_data_to_csv(file_path=Path("../results"), model_data=model_data)

#    torch.save(model.state_dict(), "../pretrained/resnet20.pth")


