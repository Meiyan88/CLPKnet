import os
import shutil,random

import numpy as np
from gen_art import Genart
path = r'D:\GRmid\npy'
out = r'G:\GliomaRecurrence\Datalist\OnlyPosOperation_First\npy_noise'

def add_noise_to_file(input_path, output_path):
    """对文件进行加噪处理并保存"""
    # 这里假设是npy文件，如果是其他格式需要修改

    for i in os.listdir(input_path):
        data_path = os.path.join(input_path,i)
        data = np.load(data_path)
        noise_intensity = random.uniform(0.01, 0.03)  # 噪声强度（根据数据范围调整）
        noise = np.random.normal(loc=0.0, scale=noise_intensity, size=data.shape).astype(
            np.float32)  # 添加高斯噪声
        #noise = Genart(data)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_path1 = os.path.join(output_path,i)
        np.save(output_path1, noise)

count = 0
for i in os.listdir(path):
    count += 1
    pati = os.path.join(path,i)
    for seq in os.listdir(pati):
        seq_file = os.path.join(pati, seq)
        out_file = seq_file.replace(r'D:\GRmid\npy', r'G:\GliomaRecurrence\Datalist\OnlyPosOperation_First\npy_noise')
        if not os.path.isdir(seq_file):
            continue

        target_folders = {'FLAIR', 'T1', 'T1C', 'T2'}

        if seq in target_folders:
            #out_file = seq_file.replace('npy','npy_noise')
            add_noise_to_file(seq_file,out_file)
        else:
            if not os.path.exists(out_file):
                shutil.copytree(seq_file,out_file)
    print('processing done:',count)


