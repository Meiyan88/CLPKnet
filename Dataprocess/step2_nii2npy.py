import SimpleITK as sitk
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def normalize(img):
    data = img
    imin = np.percentile(data, 0.1)
    imax = np.percentile(data, 99.9)
    data = ((np.clip(data, imin, imax) - imin) / (imax - imin))
    return data

# nii2npy
path = r'F:\file\zhujiang2_nii'
savepath = r'F:\file\Zhujiang_npy'
patients = os.listdir(path)
for patient in patients:
    for i in os.listdir(os.path.join(path,patient)):
        patient_i = os.path.join(patient,i)
        posra = nib.load(os.path.join(path, patient_i, "PosRA.nii.gz")).get_fdata()
        ca15 = nib.load(os.path.join(path, patient_i, "CA_15.nii.gz")).get_fdata()
        ca = nib.load(os.path.join(path, patient_i, "PosCA.nii.gz")).get_fdata()
        posra = np.flip(posra, 1)
        ca15 = np.flip(ca15, 1)
        ca = np.flip(ca, 1)
        # if len(os.listdir(os.path.join(path, patient, date))) == 5:
        flair = normalize(nib.load(os.path.join(path, patient_i, "FLAIR_registered.nii.gz")).get_fdata())
        t1 = normalize(nib.load(os.path.join(path, patient_i, "T1_registered.nii.gz")).get_fdata())
        t1c = normalize(nib.load(os.path.join(path, patient_i, "T1CE_registered.nii.gz")).get_fdata())
        t2 = normalize(nib.load(os.path.join(path, patient_i, "T2_registered.nii.gz")).get_fdata())
        # flair = normalize(nib.load(os.path.join(path, patient_i, "FLAIR.nii.gz")).get_fdata())
        # t1 = normalize(nib.load(os.path.join(path, patient_i, "T1.nii.gz")).get_fdata())
        # t2 = normalize(nib.load(os.path.join(path, patient_i, "T2.nii.gz")).get_fdata())

        # flair = np.flip(flair,1)
        # t1 = np.flip(t1,1)
        # t1c = np.flip(t1c,1)
        # t2 = np.flip(t2,1)
        #

        z_ca_min = np.min(np.where(ca15 != 0)[2])
        z_ca_max = np.max(np.where(ca15 != 0)[2])
        print('patient:{} min:{},max:{}'.format(patient,z_ca_min,z_ca_max))

        for i in range(z_ca_min, z_ca_max + 1):

            # plt.subplot(131)
            # plt.imshow(t1c[:, :, i])
            # plt.subplot(132)
            # plt.imshow(ca15[:, :, i])
            # plt.subplot(133)
            # plt.imshow(posra[:, :, i])
            # plt.close()

            if flair is not None:
                os.makedirs(os.path.join(savepath, patient, "FLAIR")) if not os.path.exists(os.path.join(savepath, patient, "FLAIR")) else None
                np.save(os.path.join(savepath, patient, "FLAIR", "data_{}.npy".format(str(i))), flair[:, :, i])
            if t1 is not None:
                os.makedirs(os.path.join(savepath, patient, "T1")) if not os.path.exists(os.path.join(savepath, patient, "T1")) else None
                np.save(os.path.join(savepath, patient, "T1", "data_{}.npy".format(str(i))), t1[:, :, i])
            if t1c is not None:
                os.makedirs(os.path.join(savepath, patient, "T1C")) if not os.path.exists(os.path.join(savepath, patient, "T1C")) else None
                np.save(os.path.join(savepath, patient, "T1C", "data_{}.npy".format(str(i))), t1c[:, :, i])
            if t2 is not None:
                os.makedirs(os.path.join(savepath, patient, "T2")) if not os.path.exists(os.path.join(savepath, patient, "T2")) else None
                np.save(os.path.join(savepath, patient, "T2", "data_{}.npy".format(str(i))), t2[:, :, i])

            os.makedirs(os.path.join(savepath, patient, "PosRA")) if not os.path.exists(os.path.join(savepath, patient, "PosRA")) else None
            # if np.sum(posra[:, :, i]) != 0:
            #     np.save(os.path.join(savepath, patient, "PosRA", "mask_{}.npy".format(str(i))), posra[:, :, i])
            np.save(os.path.join(savepath, patient, "PosRA", "mask_{}.npy".format(str(i))), posra[:, :, i])

            os.makedirs(os.path.join(savepath, patient, "CA15")) if not os.path.exists(os.path.join(savepath, patient, "CA15")) else None
            np.save(os.path.join(savepath, patient, "CA15", "mask_{}.npy".format(str(i))), ca15[:, :, i])

            os.makedirs(os.path.join(savepath, patient, "CA")) if not os.path.exists(os.path.join(savepath, patient, "CA")) else None
            np.save(os.path.join(savepath, patient, "CA", "mask_{}.npy".format(str(i))), ca[:, :, i])

        # print(patient)