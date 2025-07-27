import os,random
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import SimpleITK as sitk
from scipy.ndimage import rotate
from PIL import Image
import imageio
def Genart(img,width=random.uniform(18,28), fre=random.uniform(3, 5), fov=random.randint(5, 10)):
    # fourier transform
    #img = cv2.imread(path, 0)
    h,w = img.shape
    # if h < 256:
    #     albumentations.pad(img,min_height=256,min_width=256)
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    amp = np.abs(dft_shift)  # 振幅
    phase1 = np.angle(dft_shift)  #角度


    #
    centel = h//2
    mask = np.zeros((h, w), dtype=complex)
    mask1 = np.zeros((h, w), dtype=complex)
    for i in range(0, w):
        if centel - width < i < centel + width:
            mask[:,i] = 1
        else:

            ky = np.pi / centel * (i - centel)
            beta = random.uniform(0, np.pi/4)
            sin1 = np.sin(fre * ky + beta )  #fre * ky + beta
            phaseall = 2 * np.pi * sin1 * ky * fov#* ky * fov

            mask[:,i].real = np.array(np.cos(phaseall))
            mask[:,i].imag = np.array(np.sin(phaseall))

            #mask[i, :] = 1

    # for i in range(0, w):
    #     if centel - width < i < centel + width:
    #         mask1[:,i] = 1
    #     else:
    #
    #         ky = np.pi / centel * (i - centel)
    #         beta = random.uniform(0, np.pi/4)
    #         sin1 = np.sin(fre * ky + beta )  #fre * ky + beta
    #         phaseall = 2 * np.pi * sin1 * ky * fov#* ky * fov
    #
    #         mask1[:,i].real = np.array(np.cos(phaseall))
    #         mask1[:,i].imag = np.array(np.sin(phaseall))
    #         #mask[i, :] = 1
    #         # rot = random.randint(0,1)
    #         # if rot:
    #         #     mask1 = rotate(mask1, random.randint(0,10), reshape=False)




    # inverse transform
    real, fake = amp * np.cos(phase1), amp * np.sin(phase1)  # 实部和虚部
    spectrum = np.zeros((h, w), dtype=complex)
    spectrum.real = np.array(real)
    spectrum.imag = np.array(fake)
    spectrum1 = spectrum.copy()
    spectrum = spectrum * mask
    spectrum1 = spectrum1 * mask1

    f2shift = np.fft.ifftshift(spectrum)
    imgback = np.fft.ifft2(f2shift)
    img_tra = np.abs(imgback).astype('float32')



    return img_tra

def save_nii(img,save):
    #img = normalize_mm(img)
    out = sitk.GetImageFromArray(img)
    out.SetOrigin(out.GetOrigin())
    out.SetSpacing(out.GetSpacing())
    sitk.WriteImage(out, save)
path = r'/home/huangmeiyan/wby1/cycle2/datasets/brain1_rot/trainB'
# for i in sorted(os.listdir(path)):
#     img_path = os.path.join(path,i)
#     img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
#     img_art = Genart_rot(img)
#     img_art = np.clip(img_art,0,255.0)
#     path1 = path.replace('trainB','trainD')
#     saved = os.path.join(path1,i)
#     save_nii(img_art,saved)

