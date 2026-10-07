"""Default neural network architectures for image and tabular datasets."""

from torch import nn

class Net(nn.Module):
    """Default convolutional network used for image datasets."""

    def __init__(self, in_channels, num_classes):
        """Build the convolutional classifier."""
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Normalize the conv output to a fixed 4x4 map so the classifier accepts
        # any input size (e.g. 28x28 MedMNIST and 32x32 CIFAR-100 alike).
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, x):
        """Run the forward pass for image inputs."""
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TabularNet(nn.Module):
    """Default MLP for user-uploaded tabular custom datasets."""

    def __init__(self, in_features, num_classes):
        """Build the default multilayer perceptron for tabular inputs."""
        super().__init__()
        hidden = max(32, min(256, in_features * 2))
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        """Run the forward pass for tabular inputs."""
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)
