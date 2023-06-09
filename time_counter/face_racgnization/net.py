# -*- encoding: utf8 -*-
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 第一层卷积->激活->池化->Dropout
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2)
        )
        # 第二层卷积->激活->池化->Dropout
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )
        # 全连接层
        self.out = nn.Linear(64 * 8 * 8, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        # 对结果进行log + softmax并输出
        return nn.functional.log_softmax(x, dim=1)