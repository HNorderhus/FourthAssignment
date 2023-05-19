import torch
import torch.nn as nn

"""
This code implementation is based on the work of Aladdin Persson, https://www.youtube.com/watch?v=DkNIBBBvcPs (last seen May 18th 2023)
"""

class block(nn.Module):
    """
    A basic building block of the ResNet architecture.

    Args:
        in_channels (int): Number of input channels.
        output_channel (int): Number of output channels.
        identity_downsample (nn.Module, optional): Downsampling layer to match the shape in case of stride > 1.
        stride (int, optional): Stride value for the convolutional layers.

    Returns:
        torch.Tensor: Output tensor of the block.
    """
    def __init__(self, in_channels, output_channel, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, output_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    """
    ResNet architecture for image classification.

    Args:
        block (nn.Module): The block module to be used (e.g., `block` from above).
        layers (List[int]): List of the number of residual blocks for each layer.
        image_channels (int): Number of input channels in the image.
        num_classes (int): Number of output classes.

    Returns:
        torch.Tensor: Output tensor of the ResNet model.
    """
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block, layers[0], output_channel=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], output_channel=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], output_channel=256, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, output_channel, stride):
        identity_downsample = None
        layers = []

        # Create an identity downsample to match the shape even with the skip connection
        if stride != 1:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, output_channel, kernel_size=1, stride=stride, bias=False,),
                nn.BatchNorm2d(output_channel),
            )

        layers.append(block(self.in_channels, output_channel, identity_downsample, stride))

        # The expansion size will be replaced with the output_channels after creation of the first block and skip connection
        self.in_channels = output_channel

        # For example for second resnet layer: 64 will be mapped to 128 as output layer,
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, output_channel))

        return nn.Sequential(*layers)


def ResNet20(img_channel=3, num_classes=1000):
    return ResNet(block, [2, 3, 4], img_channel, num_classes)
    
def ResNet14(img_channel=3, num_classes=1000):
    return ResNet(block, [2, 4, 4], img_channel, num_classes)    

def ResNet8(img_channel=3, num_classes=1000):
    return ResNet(block, [1, 1, 1], img_channel, num_classes)
