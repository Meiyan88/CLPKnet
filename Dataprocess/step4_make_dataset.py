import os
import numpy as np
########################################################################################################################
# only RA and CA15 (train & validation)
data_path = r'G:\GliomaRecurrence\Datalist\OnlyPosOperation_First\npy'
#txt_path = r'/public/huangmeiyan/wby/GliomaRecurrence/Datalist/OnlyPosOperation_First/datasplit/FLAIR/'
txt_path = r'F:\file\nfzj'
#txt_path = r'/public/huangmeiyan/wby/GliomaRecurrence/Datalist/OnlyPosOperation_First/datasplit/T1_T1C_T2/'
#
for i in range(0, 5):
    train_txt = os.path.join(txt_path, "split{}".format(str(i)), "train.txt")
    test_txt = os.path.join(txt_path, "split{}".format(str(i)), "test.txt")

    train1 = open(os.path.join(txt_path, "split{}".format(str(i)), "train1.txt"), "w")
    test1 = open(os.path.join(txt_path, "split{}".format(str(i)), "test1.txt"), "w")
    with open(train_txt, "r") as lines:
        for line in lines:
            patient = line.split("\n")[0]
            print(patient)

            ra_list = os.listdir(os.path.join(data_path, patient, "PosRA"))

            for item in ra_list:
                ra_path = os.path.join(data_path, patient, "PosRA", item)
                ca_path = os.path.join(data_path, patient, "CA15", item)
                ca1_path = os.path.join(data_path, patient, "CA", item)
                #white_path = os.path.join(data_path, patient, "white_mat", item)
                ra_data = np.load(ra_path)
                if np.any(ra_data != 0):
                    train1.write(ra_path + " " + ca_path +" " + ca1_path + " "  +'\n')

    with open(test_txt, "r") as lines:
        for line in lines:
            patient = line.split("\n")[0]
            ra_list = os.listdir(os.path.join(data_path, patient, "PosRA"))
            for item in ra_list:
                ra_path = os.path.join(data_path, patient, "PosRA", item)
                ca_path = os.path.join(data_path, patient, "CA15", item)
                ca1_path = os.path.join(data_path, patient, "CA", item)
                #white_path = os.path.join(data_path, patient, "white_mat", item)
                ra_data = np.load(ra_path)
                if np.any(ra_data != 0):
                    test1.write(ra_path + " " + ca_path +" " + ca1_path + " "+'\n')

#up down only once
########################################################################################################################
#only RA and CA15 (independent test)
# data_path = r'G:\GliomaRecurrence\Datalist\OnlyPosOperation_First\npy'
# txt_path = r'G:\GliomaRecurrence\Datalist\zz2.txt'
# test1 = open(os.path.join(r'G:\GliomaRecurrence\Datalist\OnlyPosOperation_First\datasplit', "zz_sum.txt"), "w")
#
# with open(txt_path, "r") as lines:
#     for line in lines:
#         patient = line.split("\n")[0]
#
#         ca15_list = os.listdir(os.path.join(data_path, patient, "CA15"))
#
#         for item in ca15_list:
#             ca_path = os.path.join(data_path, patient, "CA15", item)
#             ra_path = os.path.join(data_path, patient, "PosRA", item)
#             ca1_path = os.path.join(data_path, patient, "CA", item)
#             if not os.path.exists(ra_path):
#                 if os.path.exists(ca_path):
#                     ca_data = np.load(ca_path)
#                     zero_data = np.zeros_like(ca_data)  # 创建相同尺寸的全零数组
#                     np.save(ra_path, zero_data)  # 保存全零数组到ra_path
#                     np.save(ca1_path, zero_data)  # 保存全零数组到ra_path
#                     print(f"Created empty RA data at {ra_path} with shape {ca_data.shape}")
#
#                 else:
#                     print(f"Warning: CA path {ca_path} does not exist, skipping")
#                     continue
#             test1.write(ra_path + " " + ca_path + " " + ca1_path + '\n')