import torch
import torchvision
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
from sympy import false

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Network.nmafa import NMaFaLayer

import torch
import torch.nn as nn
import torchvision.models as models


#2D resnet 18
class IndentityBlock(nn.Module):
    def __init__(self, in_channel, filters):
        super(IndentityBlock, self).__init__()
        F1, F2 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(in_channel, F1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            nn.Conv2d(F1, F2, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(F2)
        )
        self.relu_1 = nn.ReLU(True)

    def forward(self, X):
        X_shortcut = X
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X

class ConvBlock1(nn.Module):
    def __init__(self, in_channel, filters, s):
        super(ConvBlock1, self).__init__()
        F1, F2 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(in_channel, F1, kernel_size=(3, 3), stride=s, padding=1, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            nn.Conv2d(F1, F2, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(F2),
        )
        self.shortcut_1 = nn.Conv2d(in_channel, F2, kernel_size=(1, 1), stride=s, padding=0, bias=False)
        self.batch_1 = nn.BatchNorm2d(F2)
        self.relu_1 = nn.ReLU(True)

    def forward(self, X):
        X_shortcut = self.shortcut_1(X)
        X_shortcut = self.batch_1(X_shortcut)
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X


class ResModel18(nn.Module):
    def __init__(self, init_weights=True):
        super(ResModel18, self).__init__()
        self.Bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(True)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.stage2 = nn.Sequential(
            ConvBlock1(8, filters=[16, 16], s=(1, 1)),
            IndentityBlock(16, [16, 16])
        )
        self.stage3 = nn.Sequential(
            ConvBlock1(16, filters=[32, 32], s=(2, 2)),
            IndentityBlock(32, [32, 32])
        )
        self.stage4 = nn.Sequential(
            ConvBlock1(32, filters=[64, 64], s=(2, 2)),
            IndentityBlock(64, [64, 64])
        )
        self.stage5 = nn.Sequential(
            ConvBlock1(64, filters=[128, 128], s=(2, 2)),
            IndentityBlock(128, [128, 128])
        )

    def forward(self, X):
        out_list = []
        out = self.conv1(X)
        out = self.Bn(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out_list.append(out)

        out = self.stage2(out)
        out_list.append(out)

        out = self.stage3(out)
        out_list.append(out)

        out = self.stage4(out)
        out_list.append(out)

        out = self.stage5(out)
        out_list.append(out)

        return out_list
# 2D resnet 18

################################################################################################
# MSFEnetwork
resnet18  = models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
Moco = True
if Moco:
    resnet18 = models.resnet18()
    resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # 2. 加载预训练权重
    pretrained_dict = torch.load(r"C:\Users\61453\Desktop\MOCO299_new.pth")
    model_dict = resnet18.state_dict()

    conv1_weight_3x3 = pretrained_dict["conv1.weight"]  # 形状 [64, 1, 3, 3]
    conv1_weight_7x7 = torch.zeros(64, 1, 7, 7)  # 初始化全零 7x7 卷积核

    # 将 3x3 权重放置在 7x7 中心
    conv1_weight_7x7[:, :, 2:-2, 2:-2] = conv1_weight_3x3

    # 更新模型参数
    model_dict["conv1.weight"] = conv1_weight_7x7

    # 4. 加载调整后的权重
    resnet18.load_state_dict(model_dict, strict=False)
else:
    resnet18 = models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)


class CustomResNetEncoder(nn.Module):
    def __init__(self):
        super(CustomResNetEncoder, self).__init__()
        # 获取ResNet-18的卷积层部分（去掉全连接层）
        self.encoder = nn.Sequential(
            *list(resnet18.children())[:-2]  # 去掉最后两层（全局平均池化和全连接层）
        )

    def forward(self, x):
        return self.encoder(x)


class MSFEnet(nn.Module):
    def __init__(self, config):
        super(MSFEnet, self).__init__()


        self.encoder_flair = CustomResNetEncoder()#ResModel18()

        self.encoder_t2 = CustomResNetEncoder()#ResModel18()
        self.encoder_t1 = CustomResNetEncoder() #ResModel18()
        self.encoder_t1c = CustomResNetEncoder()#ResModel18()
        #self.encoder_prior = CustomResNetEncoder()#ResModel18()


        self.concat = False
        if self.concat:
            self.header_class = nn.Sequential(
                nn.Linear(2048, 2048, bias=False),
                nn.Dropout(p=0.4),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 1, bias=True),
            )
        else:
            self.NMaFaLayer =NMaFaLayer(model_num=4,  # change
                       in_channels=512,
                       hidden_size=128,
                       img_size=(3, 3),  ##
                       mlp_size=64,
                       self_num_layer=2,
                       window_size=(3, 3),  ##
                       token_mixer_size=4,
                       token_learner=True)

            self.header_class = nn.Sequential(
                nn.Linear(128, 128, bias=False),
                nn.Dropout(p=0.4),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1, bias=True),
            )
            self.gradients = None
            self.activations = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x,edge=None):  # x (B, C, H, W) = (B, 4, 128, 128)
        # order(flair, t2, t1, t1c)

        grad_cam = False
        flair = torch.unsqueeze(x[:, 0, :, :], dim=1)
        t2 = torch.unsqueeze(x[:, 1, :, :], dim=1)
        t1 = torch.unsqueeze(x[:, 2, :, :], dim=1)
        t1c = torch.unsqueeze(x[:, 3, :, :], dim=1)


        out_flair = self.encoder_flair(flair)  # out (B, 128, 4, 4) [-1] must match our
        out_t2 = self.encoder_t2(t2)  # out (B, 128, 4, 4)
        out_t1 = self.encoder_t1(t1)# out (B, 128, 4, 4)
        out_t1c = self.encoder_t1c(t1c)  # out (B, 128, 4, 4)
        # edge = edge.float()
        #
        # #edge = edge.squeeze(1)
        # out_prior = self.encoder_prior(edge)


        if self.concat:
            out = torch.concat((out_flair, out_t2, out_t1, out_t1c), dim=1)  # out (B, C, H, W)
            if grad_cam:
                out_attn = out
                self.activations = out_attn
                if out_attn.requires_grad:
                    h = out_attn.register_hook(self.activations_hook)
            out1 = torch.flatten(out, start_dim=2)  # out1 (B, C, 16)
            out1 = out1.permute(0, 2, 1)  # out1 (B, 16, C)
            #print(out1.shape)
            out2 = out1.contiguous().view(out1.shape[0] * 9, 512*4)  # C=32 out2 (B*16, 32)

            out3 = self.header_class(out2)  # out2 (B*16, 1)
        else:
            out = torch.cat((torch.unsqueeze(out_flair, dim=1),
                             torch.unsqueeze(out_t2, dim=1),
                             torch.unsqueeze(out_t1, dim=1),
                             torch.unsqueeze(out_t1c, dim=1),
                             #torch.unsqueeze(out_prior, dim=1),
                             ), dim=1)  # out (B, m, C, H, W) m: the number of modality
            out_attn = self.NMaFaLayer(out)  # out (B, 32, 4, 4) nxn
            if edge is not None:
                edge = edge.float()
                DFAB = False
                if DFAB:
                    print('it is DFAB')
                    out_attn = out_attn + out_attn*edge
                else:
                    out_attn = out_attn
            if grad_cam:
                self.activations = out_attn
                if out_attn.requires_grad:
                    h = out_attn.register_hook(self.activations_hook)
            b, c, _, _ = out_attn.shape
            # print(c)
            out1 = torch.flatten(out_attn, start_dim=2)  # out1 (B, C, 16)
            out1 = out1.permute(0, 2, 1)  # out1 (B, 16, C)
            out2 = out1.contiguous().view(out1.shape[0] * 9, 128)  # C=32 out2 (B*16, 32)
            out3 = self.header_class(out2)  # out2 (B*16, 1)

        return out2, out3#,out_attn


if __name__ == '__main__':
    from Configuration.PatchNet.MSFE_Class_NFtrainZJtest import config

    model = MSFEnet(config=config)
    input = torch.ones((1, 4, 96, 96))
    edge = torch.randn(1,1,96,96)
    GT = torch.ones((1, 3, 3))
    out = model(input,edge)
    GT = GT.view(1, 9)
    GT = GT.view(-1)
    print(out[1])