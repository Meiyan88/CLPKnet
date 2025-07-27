import copy
import datetime
import os, ast
import random
import re
from copy import deepcopy
import scipy.ndimage.interpolation as sci
# import cv2
# import matplotlib.pyplot as plt
# import nibabel as nib
import numpy as np
import Operation.Utils as U
import torch
import torch.utils.data as data
from Operation.Utils import toTensor
from skimage.util import random_noise
import json
# from augmentations.transforms import Compose, ElasticTransform, Rotate, RandomGamma, GaussianNoise, Flip, Resize, \
#     Normalize, PadIfNeeded, RandomCrop, RandomRotate90
from scipy import ndimage
import matplotlib.pyplot as plt
import pickle as pkl
from skimage import transform


class MSFEDataset_16(data.Dataset):
    def __init__(self, args, config, root, train_val_test):
        self.CLASSES = ['Recurr']
        self.root = root
        self.train_val_test = train_val_test
        self.config = config

        _list_f = root
        # img_path = config["image_path"]

        self.posRa = []
        self.Ca_15mm = []
        self.Ca = []

        if _list_f is not None:
            with open(_list_f, 'r') as lines:
                for line in lines:
                    _posra = line.split()[0]
                    _ca15 = line.split()[1]
                    ca = line.split()[2]

                    assert os.path.isfile(_posra), _posra + ' is not exist'
                    assert os.path.isfile(_ca15), _ca15 + ' is not exist'
                    self.posRa.append(_posra)
                    self.Ca_15mm.append(_ca15)
                    self.Ca.append(ca)

    def dataAugment(self, image, mask_posra, mask_ca15, mask_ca):
        aug_img = image.copy()
        aug_mask_posra = mask_posra.copy()
        aug_mask_ca15 = mask_ca15.copy()
        aug_mask_ca = mask_ca.copy()
        C, H, W = aug_img.shape

        #################Flip############################
        if random.random() < self.config['aug_Flip_x']:
            aug_img = np.flip(aug_img, axis=1)
            aug_mask_posra = np.flip(aug_mask_posra, axis=0)
            aug_mask_ca15 = np.flip(aug_mask_ca15, axis=0)
            aug_mask_ca = np.flip(aug_mask_ca, axis=0)

        if random.random() < self.config['aug_Flip_y']:
            aug_img = np.flip(aug_img, axis=2)
            aug_mask_posra = np.flip(aug_mask_posra, axis=1)
            aug_mask_ca15 = np.flip(aug_mask_ca15, axis=1)
            aug_mask_ca = np.flip(aug_mask_ca, axis=0)

        #################rotation#####################
        if random.random() < self.config['aug_Rotate']:
            angle = random.randint(-self.config["aug_rotate_set"], self.config["aug_rotate_set"])
            aug_img = ndimage.rotate(aug_img, angle=angle, axes=(2, 1), reshape=False, order=2, mode="constant")
            aug_mask_posra = ndimage.rotate(aug_mask_posra, angle=angle, axes=(1, 0), reshape=False, order=0,
                                            mode="constant")
            aug_mask_ca15 = ndimage.rotate(aug_mask_ca15, angle=angle, axes=(1, 0), reshape=False, order=0,
                                           mode="constant")
            aug_mask_ca = ndimage.rotate(aug_mask_ca, angle=angle, axes=(1, 0), reshape=False, order=0, mode="constant")

        if random.random() < self.config["aug_intensity_shift"]:
            for c in range(C):
                offset = random.uniform(*self.config["aug_intensity_shift_set"])
                aug_img[c] = aug_img[c] - offset

        if random.random() < self.config["aug_intensity_scale"]:
            for c in range(C):
                scale = random.uniform(*self.config["aug_intensity_scale_set"])
                aug_img[c] = aug_img[c] * scale

        # #################Gaussian#####################
        if random.random() < 0.5:
            if random.random() < self.config["aug_gaussian_noise"]:
                mean = 0
                sigma = random.uniform(0, self.config["aug_gaussian_noise_set"])
                gauss = np.random.normal(mean, sigma, aug_img.shape)
                aug_img += gauss
        else:
            if random.random() < self.config["aug_gaussian_smooth"]:
                sigma = random.uniform(0, self.config["aug_gaussian_smooth_set"])
                for c in range(C):
                    aug_img[c] = ndimage.gaussian_filter(aug_img[c], (sigma, sigma))


        return aug_img, aug_mask_posra, aug_mask_ca15, aug_mask_ca

    def __getitem__(self, index):
        #################### load data ####################
        if self.train_val_test == 'train':
            flair = np.rot90(np.load(self.posRa[index].replace("PosRA", "FLAIR").replace("mask", "data")))[np.newaxis,
                    :, :]
            t1 = np.rot90(np.load(self.posRa[index].replace("PosRA", "T1").replace("mask", "data")))[np.newaxis, :, :]
            t2 = np.rot90(np.load(self.posRa[index].replace("PosRA", "T2").replace("mask", "data")))[np.newaxis, :, :]
            t1c = np.rot90(np.load(self.posRa[index].replace("PosRA", "T1C").replace("mask", "data")))[np.newaxis, :, :]

            _mask_posra = np.rot90(np.load(self.posRa[index]))
            _mask_ca15 = np.rot90(np.load(self.Ca_15mm[index]))
            _mask_ca = np.rot90(np.load(self.Ca[index]))

            _img = np.concatenate((flair, t2, t1, t1c), axis=0)
            # print('a')

            if self.config["is_aug"]:
                if random.random() < self.config["aug_p"]:
                    # print('I am here')
                    _img, _mask_posra, _mask_ca15, _mask_ca = self.dataAugment(_img, _mask_posra, _mask_ca15, _mask_ca)

            # the union of PosRA and CA15
            C, H, W = _img.shape
            image = []
            for c in range(C):
                image.append(transform.resize(_img[c], output_shape=[256, 256], order=1, mode='constant', clip=False,
                                              preserve_range=True)[np.newaxis, :, :])
            img256 = np.concatenate(image, axis=0)

            mask_union = _mask_ca15
            ca_union = _mask_ca
            mask_union[np.where(_mask_posra != 0)] = 1
            x_min = np.min(np.where(mask_union != 0)[0])
            x_max = np.max(np.where(mask_union != 0)[0])
            y_min = np.min(np.where(mask_union != 0)[1])
            y_max = np.max(np.where(mask_union != 0)[1])

            _img = _img[:, x_min:x_max + 1, y_min:y_max + 1]

            C, H, W = _img.shape
            image = []
            for c in range(C):
                image.append(transform.resize(_img[c], output_shape=[128, 128], order=1, mode='constant', clip=False,
                                              preserve_range=True)[np.newaxis, :, :])
            _img = np.concatenate(image, axis=0)
            _img = _img.copy()
            _mask_posra = _mask_posra.copy()
            _mask_ca15 = _mask_ca15.copy()
            _mask_ca = _mask_ca15.copy()
            mask_union = mask_union[x_min:x_max + 1, y_min:y_max + 1]
            mask_gt = _mask_posra[x_min:x_max + 1, y_min:y_max + 1]
            _ca = ca_union[x_min:x_max + 1, y_min:y_max + 1]
            mask_ca = transform.resize(_ca, output_shape=[128, 128], order=0, mode='constant', clip=False,
                                       preserve_range=True, anti_aliasing=False)
            mask_gt = transform.resize(mask_gt, output_shape=[128, 128], order=0, mode='constant', clip=False,
                                       preserve_range=True, anti_aliasing=False)
            Gt = np.zeros((4, 4))

            mask_area = np.shape(np.argwhere(mask_gt != 0))[0]
            mask_area_1 = np.sum(mask_gt)

            for i in range(0, 4):
                for j in range(0, 4):
                    sub_area = np.sum(mask_gt[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32])
                    if sub_area > 0.1 * mask_area:
                        Gt[i, j] = 1
            Gt[1, 1] = 0
            _img = torch.as_tensor(_img, dtype=torch.float32)

            Gt = torch.as_tensor(Gt, dtype=torch.long)
            #  loss weight_map

            return _img, Gt

        elif self.train_val_test == 'val':
            flair = np.rot90(np.load(self.posRa[index].replace("PosRA", "FLAIR").replace("mask", "data")))[np.newaxis,
                    :, :]
            t1 = np.rot90(np.load(self.posRa[index].replace("PosRA", "T1").replace("mask", "data")))[np.newaxis, :, :]
            t2 = np.rot90(np.load(self.posRa[index].replace("PosRA", "T2").replace("mask", "data")))[np.newaxis, :, :]
            t1c = np.rot90(np.load(self.posRa[index].replace("PosRA", "T1C").replace("mask", "data")))[np.newaxis, :, :]

            _mask_posra = np.rot90(np.load(self.posRa[index]))
            _mask_ca15 = np.rot90(np.load(self.Ca_15mm[index]))

            _img = np.concatenate((flair, t2, t1, t1c), axis=0)

            # the union of PosRA and CA15
            mask_union = _mask_ca15
            mask_union[np.where(_mask_posra != 0)] = 1
            x_min = np.min(np.where(mask_union != 0)[0])
            x_max = np.max(np.where(mask_union != 0)[0])
            y_min = np.min(np.where(mask_union != 0)[1])
            y_max = np.max(np.where(mask_union != 0)[1])

            _img = _img[:, x_min:x_max + 1, y_min:y_max + 1]
            C, H, W = _img.shape
            image = []
            for c in range(C):
                image.append(transform.resize(_img[c], output_shape=[128, 128], order=1, mode='constant', clip=False,
                                              preserve_range=True)[np.newaxis, :, :])
            _img = np.concatenate(image, axis=0)
            _img = _img.copy()
            _mask_posra = _mask_posra.copy()
            _mask_ca15 = _mask_ca15.copy()
            mask_union = mask_union[x_min:x_max + 1, y_min:y_max + 1]
            mask_gt = _mask_posra[x_min:x_max + 1, y_min:y_max + 1]
            mask_union = transform.resize(mask_union, output_shape=[128, 128], order=0, mode='constant', clip=False,
                                          preserve_range=True, anti_aliasing=False)
            mask_gt = transform.resize(mask_gt, output_shape=[128, 128], order=0, mode='constant', clip=False,
                                       preserve_range=True, anti_aliasing=False)
            Gt = np.zeros((4, 4))

            mask_area = np.shape(np.argwhere(mask_gt != 0))[0]
            if mask_area == 0:
                print("assert")
            mask_area_1 = np.sum(mask_gt)

            for i in range(0, 4):
                for j in range(0, 4):
                    sub_area = np.sum(mask_gt[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32])
                    if sub_area > 0.1 * mask_area:
                        Gt[i, j] = 1
            #Gt[1, 1] = 0
            _img = torch.as_tensor(_img, dtype=torch.float32)
            # _mask_posra = torch.as_tensor(_mask_posra, dtype=torch.long)
            # _mask_ca15 = torch.as_tensor(_mask_ca15, dtype=torch.long)
            Gt = torch.as_tensor(Gt, dtype=torch.long)

            return _img, Gt

        elif self.train_val_test == 'test':
            flair = np.rot90(np.load(self.posRa[index].replace("PosRA", "FLAIR").replace("mask", "data")))[np.newaxis,
                    :, :]
            t1 = np.rot90(np.load(self.posRa[index].replace("PosRA", "T1").replace("mask", "data")))[np.newaxis, :, :]
            t2 = np.rot90(np.load(self.posRa[index].replace("PosRA", "T2").replace("mask", "data")))[np.newaxis, :, :]
            t1c = np.rot90(np.load(self.posRa[index].replace("PosRA", "T1C").replace("mask", "data")))[np.newaxis, :, :]

            _mask_posra = np.rot90(np.load(self.posRa[index]))
            _mask_ca15 = np.rot90(np.load(self.Ca_15mm[index]))
            # print(flair.shape,t1.shape,t2.shape,t1c.shape)
            if t2.shape != t1c.shape:
                #print('i am here')
                t2 = transform.resize(t2, output_shape=[1, 512, 512], order=1, mode='constant', clip=False,
                                      preserve_range=True)

            _img = np.concatenate((flair, t2, t1, t1c), axis=0)

            # the union of PosRA and CA15
            mask_union = _mask_ca15
            mask_union[np.where(_mask_posra != 0)] = 1
            x_min = np.min(np.where(mask_union != 0)[0])
            x_max = np.max(np.where(mask_union != 0)[0])
            y_min = np.min(np.where(mask_union != 0)[1])
            y_max = np.max(np.where(mask_union != 0)[1])

            _img = _img[:, x_min:x_max + 1, y_min:y_max + 1]
            C, H, W = _img.shape
            image = []
            for c in range(C):
                image.append(transform.resize(_img[c], output_shape=[128, 128], order=1, mode='constant', clip=False,
                                              preserve_range=True)[np.newaxis, :, :])
            _img = np.concatenate(image, axis=0)
            _img = _img.copy()
            _mask_posra = _mask_posra.copy()
            _mask_ca15 = _mask_ca15.copy()
            mask_union = mask_union[x_min:x_max + 1, y_min:y_max + 1]
            mask_gt = _mask_posra[x_min:x_max + 1, y_min:y_max + 1]
            mask_union = transform.resize(mask_union, output_shape=[128, 128], order=0, mode='constant', clip=False,
                                          preserve_range=True, anti_aliasing=False)
            mask_gt = transform.resize(mask_gt, output_shape=[128, 128], order=0, mode='constant', clip=False,
                                       preserve_range=True, anti_aliasing=False)
            Gt = np.zeros((4, 4))

            mask_area = np.shape(np.argwhere(mask_gt != 0))[0]
            mask_area_1 = np.sum(mask_gt)

            for i in range(0, 4):
                for j in range(0, 4):
                    sub_area = np.sum(mask_gt[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32])
                    if sub_area > 0.1 * mask_area:
                        Gt[i, j] = 1
            #Gt[1, 1] = 0
            _img = torch.as_tensor(_img, dtype=torch.float32)
            # _mask_posra = torch.as_tensor(_mask_posra, dtype=torch.long)
            # _mask_ca15 = torch.as_tensor(_mask_ca15, dtype=torch.long)
            Gt = torch.as_tensor(Gt, dtype=torch.long)

            return _img, Gt  # ,weight_map

    def __len__(self):
        return len(self.posRa)

