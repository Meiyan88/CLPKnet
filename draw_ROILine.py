import Operation.Utils as U
import torch
import numpy as np
import os
import pandas as pd
from Operation.Preparation import addArgs, getConfig, getTemp
import torch.nn.functional as F
import cv2
from skimage import transform
########################### ########################### ########################### ###########################
# Network
# from Network.PatchNetwork import PatchNet
# from Network.FLAIRNetwork import FLAIRNet
# from Network.T2Network import T2Net
# from Network.T1Network import T1Net
# from Network.T1CNetwork import T1CNet
from Network.MSFEnetwork import MSFEnet
########################### ########################### ########################### ###########################
# from Dataset.PatchDataset import PatchDataset
# from Dataset.FLAIRDataset import FLAIRDataset
# from Dataset.T2Dataset import T2Dataset
# from Dataset.T1Dataset import T1Dataset
# from Dataset.T1CDataset import T1CDataset
from Dataset.MSFEDataset import MSFEDataset

from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score,
    roc_curve,
)
from sklearn import metrics
# import matplotlib.pyplot as plt



if __name__ == '__main__':

    path = r'/public/huangmeiyan/wby/GliomaRecurrence/Datalist/OnlyPosOperation_First/datasplit/ZhuJiang_sum.txt'
    savepath = r'/public/huangmeiyan/wby/GliomaRecurrence/Save/MSFE-Resnet18_Transformer18-None-2023_02_06_17_59_21/img_result'
    with open(path, 'r') as lines:
        for line in lines:
            _posra = line.split()[0]

            flair = np.rot90(np.load(_posra.replace("PosRA", "FLAIR").replace("mask", "data"))) * 255
            t1 = np.rot90(np.load(_posra.replace("PosRA", "T1").replace("mask", "data"))) * 255
            t2 = np.rot90(np.load(_posra.replace("PosRA", "T2").replace("mask", "data"))) * 255
            t1c = np.rot90(np.load(_posra.replace("PosRA", "T1C").replace("mask", "data"))) * 255

            patient_name = _posra.split("/")[-3]
            data_num = _posra.split("/")[-1].replace("mask", "data")

            img = np.repeat(flair[:, :, np.newaxis], repeats=3, axis=-1).astype(np.uint8)

            if not os.path.exists(os.path.join(savepath, patient_name, "image")):
                os.makedirs(os.path.join(savepath, patient_name, "image"))

            cv2.imwrite(filename=os.path.join(savepath, patient_name, "image", data_num.replace("npy", "jpg")),
                        img=img)