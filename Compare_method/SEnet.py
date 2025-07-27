import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.excite = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 降维
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),  # 升维
            nn.Sigmoid()  # 输出权重
        )

    def forward(self, x):
        batch, channels, _, _ = x.size()
        y = self.squeeze(x).view(batch, channels)  # 全局池化 -> [batch, channels]
        y = self.excite(y).view(batch, channels, 1, 1)  # 生成通道权重 -> [batch, channels, 1, 1]
        return x * y  # 特征重标定


class CustomSENet(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomSENet, self).__init__()

        # 第一层卷积：支持 4 通道输入，减少下采样倍数
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # SE 块堆叠
        self.layer1 = self._make_layer(64, 64, reduction=16)
        self.layer2 = self._make_layer(64, 128, reduction=16, downsample=True)
        self.layer3 = self._make_layer(128, 256, reduction=16, downsample=True)
        self.layer4 = self._make_layer(256, 512, reduction=16, downsample=True)

        # 全局平均池化和分类头
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 1)

    def _make_layer(self, in_channels, out_channels, reduction, downsample=False):
        layers = []
        if downsample:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        else:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(SEBlock(out_channels, reduction))  # 添加 SE 模块
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
#
# model = CustomSENet(num_classes=1)
# # 创建一个随机输入张量，形状为 (1, 4, 32, 32)
# input_tensor = torch.randn(1, 4, 32, 32)
#
# # 将输入传递给模型
# output = model(input_tensor)
#
# # 打印输出形状
# print(f"Output shape: {output.shape}")