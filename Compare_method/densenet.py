import torch
import torch.nn as nn
from torchvision import models


class CustomDenseNet121(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomDenseNet121, self).__init__()

        # 加载预训练的 DenseNet-121 模型
        self.densenet = models.densenet121(pretrained=False)

        # 修改第一层卷积以支持 4 通道输入，并减少下采样倍数
        self.densenet.features.conv0 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 移除默认的最大池化层
        self.densenet.features.pool0 = nn.Identity()  # 使用 Identity 层替代 MaxPool2d

        # 修改分类头以适应自定义的类别数
        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.densenet(x)


# 创建一个随机输入张量，形状为 (1, 4, 32, 32)
input_tensor = torch.randn(288, 4, 32, 32)

model = CustomDenseNet121()
# 将输入传递给模型
output = model(input_tensor)

# 打印输出形状
print(f"Output shape: {output.shape}")