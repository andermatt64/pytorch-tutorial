import sys

import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

class CustomModule(nn.Module):
    def __init__(self):
        super(CustomModule, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 10)

    def forward(self, x):
        out = self.fc1(x)
        return out

def train(epochs=20):
    train_dataset = torchvision.datasets.MNIST(
        root='./mnist/',
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True
    )

    model = CustomModule()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    for i in range(epochs):
        for images, labels in train_loader:
            pred = model(images.view(-1, 28 * 28))

            optimizer.zero_grad()
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

        print("after {} epochs: loss = {}".format(i, loss))

    torch.save(model, "model.ckpt")

def test():
    test_dataset = torchvision.datasets.MNIST(
        root='./mnist/',
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=True
    )

    model = CustomModule()

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            out = model(images.view(-1, 28 * 28))
            if torch.argmax(out) == labels:
                correct += 1
            total += 1

        print("accuracy = {}".format(float(correct) / total))

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