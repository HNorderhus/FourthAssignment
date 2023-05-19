import torch

from src import ResNet8
from src import get_food101_dataloader
from src import train_and_evaluate_model
from src import save_training_data_to_csv
from src import plot_multiple_losses_and_accuracies

if __name__ == "__main__":
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset & Dataloader
train_loader, test_loader = get_food101_dataloader(root="./data/food-101", batch_size=64)

model = ResNet8(img_channel=3, num_classes=101)

num_epochs = 50
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 45], gamma=0.1)

train_loss, test_loss, test_top1_acc, test_top5_acc = train_and_evaluate_model(model, train_loader, test_loader, criterion,
                                                                              optimizer, num_epochs, device, scheduler)

model_data = {
    "name": "restnet8",
    "train_losses": train_loss,
    "test_losses": test_loss,
    "test_accuracies": test_top1_acc,
    "test_top5_accuracies": test_top5_acc
}

plot_multiple_losses_and_accuracies(model_data)

save_training_data_to_csv(file_path=Path("./results"), model_data=model_data)
torch.save(model.state_dict(), "./pretrained/resnet8.pth")



