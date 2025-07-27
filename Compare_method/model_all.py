from Compare_method.Convnext import ConvNeXtV2
from Compare_method.GCN_model import GCNModel
from Compare_method.SEnet import CustomSENet
from Compare_method.SwinT import SwinTransformerV2,swin_v2_tiny_4channel
from Compare_method.densenet import CustomDenseNet121
from Compare_method.HCP import HCP

def compare_model(id):
    if id == 1:
        model = HCP()
    if id == 2:
        model = CustomDenseNet121()
    if id == 3:
        model = CustomSENet()
    if id == 4:
        model =ConvNeXtV2()
    if id == 5:
        model = swin_v2_tiny_4channel()
    if id == 6:
        model = GCNModel()

    return model

