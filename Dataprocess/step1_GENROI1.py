import numpy as np
import os
import SimpleITK as itk
import skimage.transform as st
from shutil import copyfile
from skimage.util import crop
import nibabel as nib
import math
import pandas as pd
from skimage.morphology import  opening, dilation, disk, cube, ball
from skimage.morphology import remove_small_objects,remove_small_holes,flood_fill,binary_closing
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import shutil
from skimage.morphology import footprint_rectangle
def step4(mask, fill=1):
    img = flood_fill(mask,seed_point=(0,0,0), new_value=fill)   # label for fill
    img = flood_fill(img, seed_point=(0, img.shape[1]-1, 0), new_value=fill)  # label for fill
    img = flood_fill(img, seed_point=(0, 0, img.shape[2]-1), new_value=fill)  # label for fill
    img = flood_fill(img,seed_point=(img.shape[0]-1,0,0), new_value=fill)   # label for fill
    img = flood_fill(img, seed_point=(img.shape[0]-1, img.shape[1]-1, img.shape[2]-1), new_value=fill)
    img = flood_fill(img, seed_point=(0, img.shape[1] - 1, img.shape[2] - 1), new_value=fill)
    img = flood_fill(img, seed_point=(img.shape[0] - 1, 0, img.shape[2] - 1), new_value=fill)
    img = flood_fill(img, seed_point=(img.shape[0] - 1, img.shape[1] - 1, 0), new_value=fill)
    img = np.where(img == 0, fill, mask)
    return img

def fill2d(mask, fill=1):
    img = flood_fill(mask,seed_point=(0, 0), new_value=fill)   # label for fill
    img = flood_fill(img, seed_point=(0, img.shape[1]-1), new_value=fill)  # label for fill
    img = flood_fill(img,seed_point=(img.shape[0]-1,0), new_value=fill)   # label for fill
    img = flood_fill(img, seed_point=(img.shape[0]-1, img.shape[1]-1), new_value=fill)
    img = np.where(img == 0, fill, mask)
    return img


if __name__ == '__main__':
    ##1.5cm
    path = r"F:\file\Zhujiang"
    file = os.listdir(path)
    denoise = footprint_rectangle((1, 1, 1))
    outsize = 15
    for sample_name in os.listdir(path):
        sample_path = os.path.join(path, sample_name)

        # 跳过非目录项
        if not os.path.isdir(sample_path):
            continue

        # 获取并排序时间点子目录
        time_subdirs = [d for d in os.listdir(sample_path)
                        if os.path.isdir(os.path.join(sample_path, d))]
        first_time_subdir = sorted(time_subdirs)[0]
        img_path = os.path.join(sample_path, first_time_subdir, 'PosCA.nii.gz')

        if not os.path.exists(img_path):
            print(f"跳过样本 {sample_name} (缺少PosRA.nii.gz文件)")
            continue

        if os.path.exists(os.path.join(sample_path, first_time_subdir,'CA_15.nii.gz')):
            print(f"跳过样本 {sample_name} (已经有PosRA.nii.gz文件)")
            continue

        img = itk.ReadImage(img_path)
        space = img.GetSpacing()
        direction = img.GetDirection()
        origin = img.GetOrigin()
        x = itk.GetArrayFromImage(img)

        xy = int(outsize // space[0])
        z = int(outsize // space[2])
        StanderKernel = ball(xy)
        slicez1 = (StanderKernel.shape[0] - 1) // 2 - round(
            (space[2] / outsize) * ((StanderKernel.shape[0] - 1) // 2))
        slicez0 = (StanderKernel.shape[0] - 1) // 2 - round(
            (z * space[2] / outsize) * ((StanderKernel.shape[0] - 1) // 2))
        slicez2 = (StanderKernel.shape[0] - 1) // 2
        slicez3 = (StanderKernel.shape[0] - 1) - slicez1
        slicez4 = (StanderKernel.shape[0] - 1) - slicez0
        index = np.asarray([slicez0, slicez1, slicez2, slicez3, slicez4], dtype=int)
        adjust_kernel = StanderKernel[index]

        x = opening(x, denoise)
        x = dilation(x, adjust_kernel)

        file = itk.GetImageFromArray(x.astype(np.float32))
        file.SetSpacing(space)
        file.SetOrigin(origin)
        file.SetDirection(direction)
        itk.WriteImage(file,
                       os.path.join(sample_path, first_time_subdir, 'CA_15.nii.gz'))
        print(sample_path, first_time_subdir,'CA_15.nii.gz')