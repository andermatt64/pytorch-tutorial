import sys

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

def show_image(x):
    plt.imshow(np.transpose(x.numpy(), (1, 2, 0)))
    plt.show()

class CustomModule(nn.Module):
    def __init__(self):
        super(CustomModule, self).__init__()

    def forward(self, x):
        pass

def train():
    train_dataset = torchvision.datasets.CIFAR10(
        root='./cifar10/',
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True
    )

def test():
    test_dataset = torchvision.datasets.CIFAR10(
        root='./cifar10/',
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: {} (train|test)".format(sys.argv[0]))
        sys.exit(1)

    action = sys.argv[1].lower().replace(" ", "")
    if action == "train":
        train()
    elif action == "test":
        test()
    else:
        print("Invalid option: {}".format(action))