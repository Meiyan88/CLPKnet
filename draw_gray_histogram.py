import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def normalize(img):
    data = img
    imin = np.percentile(data, 0.1)
    imax = np.percentile(data, 99.9)
    data = ((np.clip(data, imin, imax) - imin) * 255 / (imax - imin))
    return data

if __name__ == '__main__':
    path = r'/public/huangmeiyan/wby/GliomaRecurrence/Datalist/OnlyPosOperation_First/nii/'
    savepath = r'/public/huangmeiyan/wby/GliomaRecurrence/Gray_histogram1/train1'
    patients = os.listdir(path)
    txt_path = r'/public/huangmeiyan/wby/GliomaRecurrence/Datalist/OnlyPosOperation_First/datasplit/FLAIR/split0/train.txt'
    with open(txt_path, 'r') as file:
        # 读取文件内容并将每一行放入一个新的列表中
        lines = [line.strip() for line in file]

    print(lines)
    print(len(lines),len(patients))
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    print(patients)
    models = ["T1C_AX.nii.gz", "T1.nii.gz", "FLAIR.nii.gz", "T2.nii.gz"]
    for model in models:
        roi_gray_list = []
        recurr_gray_list = []
        for patient in patients:
            date = sorted(os.listdir(os.path.join(path, patient)))[0]
            sequences = os.listdir(os.path.join(path, patient, date))
            if len(sequences) == 4:
                data = nib.load(os.path.join(path, patient, date, model)).get_fdata()
                data = normalize(data)
                roi = nib.load(os.path.join(path, patient, "draw", "CA_15mm.nii.gz")).get_fdata()
                recurr = nib.load(os.path.join(path, patient, "draw", "PosRA.nii.gz")).get_fdata()

                roi[np.where(recurr == 1)] = 0
                data_roi = data * roi
                data_recurr = data * recurr

                arr_roi = data_roi.flatten()
                arr_recurr = data_recurr.flatten()
                roi_gray_list.append(arr_roi)
                recurr_gray_list.append(arr_recurr)
                print(model, patient)

        roi_gray_list = np.concatenate((roi_gray_list))
        recurr_gray_list = np.concatenate((recurr_gray_list))

        plt.figure()
        n4, bins4, patches4 = plt.hist(x=roi_gray_list, bins=range(1, 255), facecolor='blue', alpha=0.75, density=True, stacked=True)
        n5, bins5, patches5 = plt.hist(x=recurr_gray_list, bins=range(1, 255), facecolor='red', alpha=0.75, density=True, stacked=True)
        # plt.savefig(os.path.join(savepath, model.split('.')[0] + '.jpg'))
        # plt.close()