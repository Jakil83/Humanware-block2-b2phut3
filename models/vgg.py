"""VGG11/13/16/19 in Pytorch.

Source: https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py
"""
import torch
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes_length, num_classes_digits):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier_length = nn.Linear(512, num_classes_length)
        self.classifier_digit1 = nn.Linear(512, num_classes_digits)
        self.classifier_digit2 = nn.Linear(512, num_classes_digits)
        self.classifier_digit3 = nn.Linear(512, num_classes_digits)
        self.classifier_digit4 = nn.Linear(512, num_classes_digits)
        self.classifier_digit5 = nn.Linear(512, num_classes_digits)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return [self.classifier_length(out), self.classifier_digit1(out), self.classifier_digit2(out),
                self.classifier_digit3(out), self.classifier_digit4(out), self.classifier_digit5(out)]

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG19', num_classes_digits=10, num_classes_length=7)
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y)

