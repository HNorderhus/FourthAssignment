import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from src import ResNet8, ResNet20
from src import get_food101_dataloader
import torchvision.transforms as transforms
from PIL import Image

def visualize_results(model, image, class_names, device=torch.device('cuda')):
    """
    Visualizes the results of a neural network by plotting the image with its predictions.

    Args:
        model (nn.Module): The trained neural network model.
        image (torch.Tensor): The input image tensor.
        class_names (list): A list of class names corresponding to the dataset class indices.
        device (torch.device): The device on which the model is running (e.g., 'cuda' or 'cpu').
    """
    model.to(device)  
    model.eval()  

    # Function to unnormalize and convert the tensor to a numpy array
    def unnormalize(tensor):
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        return (tensor.numpy() * std[:, None, None] + mean[:, None, None]).transpose((1, 2, 0))


    image = image.to(device)  # Move the image tensor to the specified device

    with torch.no_grad():
        output = model(image)
        probabilities, preds = torch.topk(torch.nn.functional.softmax(output, dim=1), k=5, dim=1)

        plt.imshow(unnormalize(image.squeeze(0).cpu()))
        plt.axis('off')

        pred_text = "Predictions:\n"
        pred_classes = [class_names[p.item()] for p in preds[0]]  # Convert tensor to list of class names
        pred_probs = [prob.item() * 100 for prob in probabilities[0]]  # Convert tensor to list of probabilities

        for j in range(5):
            pred_class = pred_classes[j]
            pred_prob = pred_probs[j]

            pred_text += f"{pred_class}: {pred_prob:.2f}%\n"

        plt.title(f"{pred_text}")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Food Recognition Inference')
    parser.add_argument('--model', type=str, choices=['resnet8', 'resnet20'], default='resnet8',
                        help='Model variation (default: resnet8)')
    parser.add_argument('--weights', type=str, required=True, help='Path to pretrained weights file')
    parser.add_argument('--image', type=str, required=True, help='Path to the inference image')
    parser.add_argument('--class_file', type=str, required=True, help='Path to the class file')
    args = parser.parse_args()

    with open(args.class_file, 'r') as f:
        classes = [line.strip() for line in f]

    if args.model == "resnet8":
        model = ResNet8(img_channel=3, num_classes=101)
    elif args.model =="resnet14":
      	model = ResNet14(img_channel=3, num_classes=101)
    else:
        model = ResNet20(img_channel=3, num_classes=101)  

    weights = torch.load(args.weights)
    model.load_state_dict(weights)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load the inference image
    image = Image.open(args.image).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    train_loader, test_loader = get_food101_dataloader(root="./data", batch_size=64)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    image_tensor = image_tensor.to(device)

    visualize_results(model, image_tensor, classes)
