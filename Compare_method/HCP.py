import torch.nn as nn
import torch.nn.functional as F


class HCP(nn.Module):
    def __init__(self):
        super(HCP, self).__init__()
        # 输入通道改为1
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1)  # 96x96
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 32x32
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # 16x16
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 8x8

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # 8x8
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 4x4

        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1)  # 4x4
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)  # 3x3

        # 全连接层适配新尺寸
        self.fc1 = nn.Linear(512 , 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # (1,64,96,96)
        x = self.pool1(x)  # (1,64,48,48)

        x = F.relu(self.conv2(x))  # (1,128,48,48)
        x = self.pool2(x)  # (1,128,24,24)

        x = F.relu(self.conv3(x))  # (1,256,24,24)
        x = self.pool3(x)  # (1,256,12,12)

        x = F.relu(self.conv4(x))  # (1,512,12,12)
        x = self.pool4(x)  # (1,512,6,6)

        x = F.relu(self.conv5(x))  # (1,512,6,6)
        x = self.pool5(x)  # (1,512,3,3) 中间层输出

        x = x.view(x.size(0),x.size(1), -1)  # 展平为 (1, 512*3*3)
        x = x.permute(0, 2, 1)  # out1 (B, 16, C)
        x = x.contiguous().view(x.shape[0] * 9, 512)  # C=32 out2 (B*16, 32)


        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        return x
# import torch.nn as nn
#
#
# class HCP(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         # 初始卷积层（保持尺寸）
#         self.conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1)  # 输入: (1,4,96,96) → 输出: (1,64,96,96)
#
#         # 用 stride=2 的卷积替代 pooling（下采样至 48x48）
#         self.downsample1 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)  # (1,64,48,48)
#
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # (1,128,48,48)
#         self.downsample2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)  # (1,128,24,24)
#
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # (1,256,24,24)
#         self.downsample3 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)  # (1,256,12,12)
#
#         self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # (1,512,12,12)
#         self.downsample4 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)  # (1,512,6,6)
#
#         self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1)  # (1,512,6,6)
#         self.downsample5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)  # (1,512,3,3)
#
#         # 全连接层（适配最终尺寸 3x3）
#         self.fc1 = nn.Linear(512 * 3 * 3, 128)  # 输入展平为 512*3*3
#         self.dropout1 = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(128, 1)
#         self.dropout2 = nn.Dropout(0.5)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.downsample1(x)
#
#         x = self.conv2(x)
#         x = self.downsample2(x)
#
#         x = self.conv3(x)
#         x = self.downsample3(x)
#
#         x = self.conv4(x)
#         x = self.downsample4(x)
#
#         x = self.conv5(x)
#         x = self.downsample5(x)
#
#         # 展平后输入全连接层
#         x = x.view(x.size(0), -1)  # (batch_size, 512*3*3)
#         x = self.dropout1(self.fc1(x))
#         x = self.dropout2(self.fc2(x))
#         return x

# 实例化并验证
# model = HCP()
# input_tensor = torch.randn(1, 4, 96, 96)
# output = model(input_tensor)
#
# out = model(input_tensor)
# print(out.shape)