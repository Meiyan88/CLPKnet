from sklearn.model_selection import KFold
import os
import numpy as np

# #split 5 fold(Kfold)
# path = r'F:\file\npy\zhongzhong_npy'
# path1 = r'F:\file\npy\zhongzhong_npy'
save_path = r'F:\file\nfzj'
ZhuJiang_txt = r'F:\file\zhujiang1.txt'
#patients = os.listdir(path)
zhujiang = []
txt =  open(ZhuJiang_txt, "r")

# for i in os.listdir(path1):
#     txt.write(i+ '\n')
# Nanfang = []
# for patient in patients:
#         date = sorted(os.listdir(os.path.join(path, patient)))[0]
#
#         Nanfang.append(patient)

skf = KFold(n_splits=5, shuffle=True, random_state=10)
train_data = []
test_data = []
i = 0
Nanfang = []
with open(ZhuJiang_txt, "r", encoding="utf-8") as file:
    for line in file:
        Nanfang.append(line.strip())  # 使用strip()移除每行首尾的空白符（如换行符）

for train_index, test_index in skf.split(Nanfang):
    if not os.path.exists(os.path.join(save_path, 'split'+str(i))):
        os.makedirs(os.path.join(save_path, 'split'+str(i)))

    train_path = os.path.join(save_path, 'split'+str(i), 'train.txt')
    test_path = os.path.join(save_path, 'split'+str(i), 'test.txt')

    train_txt = open(train_path, 'w')
    test_txt = open(test_path, 'w')

    for j in train_index:
        train_txt.write(str(Nanfang[j]) + '\n')
    for k in test_index:
        test_txt.write(str(Nanfang[k]) + '\n')
    i = i + 1

    