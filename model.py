import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()

        # in_channels = 1 since Mel-Spectograms are gonna be ~128x128x1
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = self.make_layer(64, 64, 2)
        self.conv3 = self.make_layer(64, 128, 2, stride=2)
        self.conv4 = self.make_layer(128, 256, 2, stride=2)
        self.conv5 = self.make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # first layer forward pass
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        # pass through all residual blocks
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # global pooling and fc layer at the end
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride, downsample))

        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels))

        return nn.Sequential(*layers)


class MLP(nn.Module):
    # architecture: {linear-bn-relu-dropout}* -> linear
    def __init__(self, input_dim=58, hidden_dims: list[int] = [], num_classes=10):
        super().__init__()
        self.model = self.make_model(input_dim, hidden_dims, num_classes)
        self.initialize_weights()

    def forward(self, x):
        return self.model(x)

    def make_model(self, input_dim, hidden_dims, num_classes, dropout_p=0.2):
        layers = []

        for in_dim, h_dim in zip([input_dim, *hidden_dims[:-1]], hidden_dims):
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_p)
            ])

        if hidden_dims:
            layers.append(nn.Linear(hidden_dims[-1], num_classes))
        else:
            layers.append(nn.Linear(input_dim, num_classes))

        return nn.Sequential(*layers)

    def initialize_weights(self):
        # kaiming init for all linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
