import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from Dataset.MSFEDataset import MSFEDataset
from Network.MSFEnetwork import MSFEnet
import Operation.Utils as U
import numpy as np
import os
import pandas as pd
from Operation.Preparation import addArgs, getConfig, getTemp
from PIL import Image
from skimage.transform import resize
def compute_nmafa_gradcam(model, input_tensor):
    """
    修改后的Grad-CAM计算函数（适配二分类输出）
    """
    # 前向传播
    _, output, attn_output = model(input_tensor)

    # 二分类处理（输出形状应为 [B, 1]）
    output = output.squeeze()  # 变为 [B]
    model.zero_grad()
    output.backward(torch.ones_like(output), retain_graph=True)  # 计算梯度

    # 获取梯度和激活值
    gradients = model.gradients  # [B, C, H, W]
    activations = model.activations  # [B, C, H, W]

    # 计算权重（全局平均梯度）
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])  # [C]

    # 加权激活图
    weighted_activations = activations * pooled_gradients[None, :, None, None]
    heatmap = torch.mean(weighted_activations, dim=1).squeeze()  # [H, W]

    # ReLU和归一化
    heatmap = torch.relu(heatmap)
    heatmap /= torch.max(heatmap) + 1e-10  # 避免除零

    return heatmap.detach().cpu().numpy(), attn_output.detach().cpu().numpy()


def visualize_nmafa_gradcam(input_image, heatmap, saved, modality_idx=0):
    """
    可视化NMaFaLayer后的Grad-CAM
    """
    # 准备输入图像
    img = input_image[:, modality_idx, :, :].squeeze().cpu().numpy()
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    # 调整热力图大小
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # 创建可视化
    fig, axes = plt.subplots(1, 2, figsize=(5, 5))

    # 原始图像
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title(f"原始图像 (模态{modality_idx})")
    axes[0].axis('off')

    # Grad-CAM热力图
    axes[1].imshow(img, cmap='gray')
    axes[1].imshow(heatmap_resized, cmap='jet', alpha=0.7)
    axes[1].set_title("Grad-CAM热力图")
    axes[1].axis('off')

    # 注意力输出可视化 (取第一个通道)
    # attn_vis = attn_output[0, 0, :, :]
    # axes[2].imshow(attn_vis, cmap='jet')
    # axes[2].set_title("NMaFaLayer输出")
    # axes[2].axis('off')

    # plt.tight_layout()
    # plt.show()
    # plt.save(saved, bbox_inches='tight', dpi=300)
    # plt.close()  # 关闭图像，避免重复显示

def save_heatmap(input_image,heatmap, saved,modality_idx=3):
    img = input_image[:, modality_idx, :, :].squeeze().cpu().numpy()
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    # 调整热力图大小
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    # 1. 创建一个新的空白figure（不显示窗口）
    fig = plt.figure(figsize=(5, 5), dpi=300)  # 设置高分辨率
    ax = fig.add_axes([0, 0, 1, 1])  # 覆盖整个画布
    original_heatmap = heatmap_resized

    # 1. 设置缩小比例（0.5表示缩小到50%）
    scale_factor = 0.5

    # 2. 使用scikit-image的高质量缩放（抗锯齿）
    resized_heatmap = resize(original_heatmap,
                             (int(original_heatmap.shape[0] * scale_factor),
                              int(original_heatmap.shape[1] * scale_factor)),
                             anti_aliasing=True)

    # 3. 创建图形时调整figsize（按比例缩小）
    fig = plt.figure(figsize=(8 * scale_factor, 8 * scale_factor), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])

    # 4. 显示图像（如果需要显示原图+热力图）
    ax.imshow(img, cmap='gray')
    ax.imshow(resized_heatmap, cmap='jet', alpha=0.7,
              extent=(0, img.shape[1], img.shape[0], 0))  # extent保持坐标对齐
    ax.axis('off')

    # 5. 保存（保持300dpi高清）
    plt.savefig(saved,
                bbox_inches='tight',
                pad_inches=0,
                dpi=300)
    plt.close()

if __name__ == '__main__':
    from Configuration.PatchNet.ConcatC_Class_NFtrainZJtest import config
    torch.set_num_threads(8)
    args = addArgs()
    savepath = r'G:\GliomaRecurrence\Save\CLPKnet\KF3\heatmap'
    #patient_name = posra.split("/")[-3]
    model_path = r'G:\GliomaRecurrence\Save\CLPKnet\KF3\epoch_38.pth'

    test_data_path = r'G:\GliomaRecurrence\Datalist\OnlyPosOperation_First\datasplit\ZhuJiang_sum.txt'#config["independent_dataset"]
    #savepath = r'C:\Users\61453\Desktop\GliomaRecurrence\Save\Multi_center\img_result'
    #savepath = model_path

    dataset = MSFEDataset(args, config, root=test_data_path, train_val_test='test')
    test_dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    sum_batchsize = len(test_dataset_loader)

    network = MSFEnet(config=config)
    network.load_state_dict(torch.load(model_path, map_location='cpu'))

    #network.cuda(0)
    network.eval()
    #images0 = r''
    for images, labels,address in test_dataset_loader:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        patient_name = address[0].split('/')[-3]
        idx  =  address[0].split('/')[-1].replace('.npy','.png')
        if not os.path.exists(os.path.join(savepath, patient_name)):
            os.makedirs(os.path.join(savepath, patient_name))


        images.to(device)
        labels.to(device)
        heatmap, attn_output =  compute_nmafa_gradcam(network,images)
        saved= os.path.join(savepath, patient_name, idx)
        save_heatmap(images,heatmap,saved=saved,modality_idx=0)


        # img_np = images[0].permute(1, 2, 0).cpu().numpy()  # [H,W,C]
        # if img_np.shape[2] == 4:
        #     img_np = img_np[:, :, :3]
        # img_np = (img_np * 255).astype(np.uint8)  # 转为0-255
        #
        # # 处理热力图
        # heatmap = (heatmap * 255).astype(np.uint8)
        # heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        # heatmap_colored = cv2.resize(heatmap_colored, (img_np.shape[1], img_np.shape[0]))
        # combine = np.hstack([img[0], heatmap])
        # heatmap_normalized = (heatmap * 255).astype(np.uint8)
        # saved= os.path.join(savepath, patient_name, idx)
        # heatmap_pil = Image.fromarray(heatmap_normalized)  # 'L' 表示灰度图
        # heatmap_pil.save(saved)


        #visualize_nmafa_gradcam(images, heatmap, attn_output)



