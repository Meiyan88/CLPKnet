import torch.nn as nn
import torch
from einops import rearrange
from Network.multi_spatial_att import MultiSpatialFusion
from Network.cross_modality_att import CrossModalityFusion

class NMaFaLayer(nn.Module):
    def __init__(self, model_num,
                 in_channels,
                 hidden_size,
                 img_size,
                 mlp_size=256,
                 self_num_layer=2,
                 window_size=(4, 4),
                 token_mixer_size=32,
                 token_learner=False):
        super().__init__()
        self.img_size = img_size
        self.hidden_size = hidden_size
        # self.conv1 = nn.Conv2d(2048,512,kernel_size=1)
        # self.conv2 = nn.Conv2d(512,128,kernel_size=1)

        self.spatial_att = MultiSpatialFusion(in_channels=model_num*in_channels,
                                                hidden_size=hidden_size,
                                                img_size=img_size,
                                                mlp_size=mlp_size,
                                                num_layers=self_num_layer,
                                                window_size=window_size)

        self.modality_att = CrossModalityFusion(model_num=model_num,
                                               in_channels=in_channels,
                                               hidden_size=hidden_size,
                                               img_size=img_size,
                                               mlp_size=mlp_size,
                                               token_mixer_size=token_mixer_size,
                                               token_learner=token_learner)

    def forward(self, x):
        # x: (batch, modal_num, hidden_size, d, w, h)
        q = rearrange(x, "b m f w h -> b (m f) w h")
        # q = self.conv1(q)
        # q = self.conv2(q)
        q = self.spatial_att(q) # b c w h
        fusion_out = self.modality_att(q, x)
        return fusion_out


if __name__ == '__main__':
    # t1 = torch.rand(1, 4, 512, 4, 4)
    #
    # model = NMaFaLayer(model_num=4,
    #              in_channels=512,
    #              hidden_size=128,
    #              img_size=(4, 4),
    #              mlp_size=256,
    #              self_num_layer=2,
    #              window_size=(2, 2),
    #              token_mixer_size=4,
    #              token_learner=True)

    t1 = torch.rand(8, 4, 512, 2, 2)

    model = NMaFaLayer(model_num=4,  # change
                                 in_channels=512,
                                 hidden_size=128,
                                 img_size=(2, 2),  ##
                                 mlp_size=64,
                                 self_num_layer=2,
                                 window_size=(2, 2),  ##
                                 token_mixer_size=4,
                                 token_learner=True)

    out = model(t1)
    print(out.shape)
