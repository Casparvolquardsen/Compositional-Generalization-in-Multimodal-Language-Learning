import torch
from torch import nn
from torchvision import models
from torchvision import transforms
from collections import OrderedDict
from einops.layers.torch import Rearrange


class EisermannVGG(nn.Module):
    def __init__(self, out_features=32, dropout2=0.5, freeze=True):
        super().__init__()

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224)])

        self.vgg16_features = models.vgg16(pretrained=True).features
        # self.dropout1 = nn.Dropout(p=dropout1, inplace=False)
        self.fc = nn.Linear(in_features=25088, out_features=out_features, bias=True)
        self.dropout2 = nn.Dropout(p=dropout2, inplace=False)

        if freeze:
            self.vgg16_features.requires_grad_(False)

    def forward(self, x):
        x = self.transform(x)
        x = self.vgg16_features(x)
        x = torch.flatten(x, 1)
        # x = self.dropout1(x)
        x = self.fc(x)
        x = self.dropout2(x)

        return x


class ResNet18(nn.Module):
    def __init__(self, pretrained=False, convolutional_features=1024, out_features=256, dropout1=0.0, dropout2=0.0,
                 freeze=False):
        super().__init__()

        self.resnet = nn.Sequential(OrderedDict([
            ("conv1", models.resnet18(pretrained=pretrained, ).conv1),
            ("bn1", models.resnet18(pretrained=pretrained).bn1),
            ("relu", nn.ReLU(inplace=True)),
            ("maxpool", nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)),
            ("layer1", models.resnet18(pretrained=pretrained).layer1),
            ("layer2", models.resnet18(pretrained=pretrained).layer2),
            ("layer3", models.resnet18(pretrained=pretrained).layer3),
            ("layer4", models.resnet18(pretrained=pretrained).layer4),  # TODO freeze not
            ("avgpool", nn.AdaptiveAvgPool2d(output_size=(1, 2 if convolutional_features == 1024 else 1))),
            ("flatten", Rearrange('b c w h -> b (c w h)')),
            ("dropout1", nn.Dropout(p=dropout1)),
            ("fc", nn.Linear(in_features=convolutional_features, out_features=out_features, bias=True)),
            ("dropout2", nn.Dropout(p=dropout2))
        ]))

        if freeze:
            for name, param in self.resnet.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False

    def forward(self, x):
        x = models.resnet18(pretrained=True, ).conv1(x)
        x = models.resnet18(pretrained=True).bn1(x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)(x)
        x = models.resnet18(pretrained=True).layer1(x)

        return self.resnet(x)


class ResNet34(nn.Module):
    def __init__(self, pretrained=False, convolutional_features=1024, out_features=256, dropout1=0.0, dropout2=0.0,
                 freeze=False):
        super().__init__()

        self.resnet = nn.Sequential(OrderedDict([
            ("conv1", models.resnet34(pretrained=pretrained).conv1),
            ("bn1", models.resnet34(pretrained=pretrained).bn1),
            ("relu", nn.ReLU(inplace=True)),
            ("maxpool", nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)),
            ("layer1", models.resnet34(pretrained=pretrained).layer1),
            ("layer2", models.resnet34(pretrained=pretrained).layer2),
            ("layer3", models.resnet34(pretrained=pretrained).layer3),
            ("layer4", models.resnet34(pretrained=pretrained).layer4),
            ("avgpool", nn.AdaptiveAvgPool2d(output_size=(1, 2 if convolutional_features == 1024 else 1))),
            ("flatten", Rearrange('b c w h -> b (c w h)')),
            ("dropout1", nn.Dropout(p=dropout1)),
            ("fc", nn.Linear(in_features=convolutional_features, out_features=out_features, bias=True)),
            ("dropout2", nn.Dropout(p=dropout2))
        ]))

        if freeze:
            for name, param in self.resnet.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False

    def forward(self, x):
        return self.resnet(x)


class ResNet50(nn.Module):
    def __init__(self, pretrained=False, convolutional_features=1024, out_features=256, dropout1=0.0, dropout2=0.0,
                 freeze=False):
        super().__init__()

        self.resnet = nn.Sequential(OrderedDict([
            ("conv1", models.resnet50(pretrained=pretrained).conv1),
            ("bn1", models.resnet50(pretrained=pretrained).bn1),
            ("relu", nn.ReLU(inplace=True)),
            ("maxpool", nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)),
            ("layer1", models.resnet50(pretrained=pretrained).layer1),
            ("layer2", models.resnet50(pretrained=pretrained).layer2),
            ("layer3", models.resnet50(pretrained=pretrained).layer3),
            ("layer4", models.resnet50(pretrained=pretrained).layer4),
            ("avgpool", nn.AdaptiveAvgPool2d(output_size=(1, 2 if convolutional_features == 4096 else 1))),
            ("flatten", Rearrange('b c w h -> b (c w h)')),
            ("dropout1", nn.Dropout(p=dropout1)),
            ("fc", nn.Linear(in_features=convolutional_features, out_features=out_features, bias=True)),
            ("dropout2", nn.Dropout(p=dropout2))
        ]))

        if freeze:
            for name, param in self.resnet.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False

    def forward(self, x):
        return self.resnet(x)


class ResNet101(nn.Module):
    def __init__(self, pretrained=False, convolutional_features=1024, out_features=256, dropout1=0.0, dropout2=0.0,
                 freeze=False):
        super().__init__()

        self.resnet = nn.Sequential(OrderedDict([
            ("conv1", models.resnet101(pretrained=pretrained).conv1),
            ("bn1", models.resnet101(pretrained=pretrained).bn1),
            ("relu", nn.ReLU(inplace=True)),
            ("maxpool", nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)),
            ("layer1", models.resnet101(pretrained=pretrained).layer1),
            ("layer2", models.resnet101(pretrained=pretrained).layer2),
            ("layer3", models.resnet101(pretrained=pretrained).layer3),
            ("layer4", models.resnet101(pretrained=pretrained).layer4),
            ("avgpool", nn.AdaptiveAvgPool2d(output_size=(1, 2 if convolutional_features == 4096 else 1))),
            ("flatten", Rearrange('b c w h -> b (c w h)')),
            ("dropout1", nn.Dropout(p=dropout1)),
            ("fc", nn.Linear(in_features=convolutional_features, out_features=out_features, bias=True)),
            ("dropout2", nn.Dropout(p=dropout2))
        ]))

        if freeze:
            for name, param in self.resnet.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False

    def forward(self, x):
        return self.resnet(x)


if __name__ == "__main__":
    resnet18 = ResNet18(pretrained=True)
    print(resnet18)

    test = torch.zeros((1, 3, 224, 398))
    out = resnet18(test)
    print(out)
