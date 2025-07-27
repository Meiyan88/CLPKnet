import cv2
import numpy as np

img_path = r'G:\GliomaRecurrence\Datalist\OnlyPosOperation_First\npy\LinHongYin2210887\T1C\data_25.npy'
mask15_path = r'G:\GliomaRecurrence\Datalist\OnlyPosOperation_First\npy\LinHongYin2210887\CA15\mask_25.npy'
mask_path = r'G:\GliomaRecurrence\Datalist\OnlyPosOperation_First\npy\LinHongYin2210887\PosRA\mask_25.npy'
mask_union = np.rot90(np.load(mask15_path))
mask = np.rot90(np.load(mask_path))
img = np.rot90(np.load(img_path))
x_min = np.min(np.where(mask_union != 0)[0])
x_max = np.max(np.where(mask_union != 0)[0])
y_min = np.min(np.where(mask_union != 0)[1])
y_max = np.max(np.where(mask_union != 0)[1])
# 假设 img 是 [H,W,C] 的 numpy 数组（OpenCV 格式：BGR）
if img.dtype != np.uint8:
    img = (img * 255).astype(np.uint8)

img_with_box = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

box_mask = np.zeros_like(mask, dtype=np.uint8)
box_mask[x_min:x_max+1, y_min:y_max+1] = 1  # 框内区域设为1
label = mask * box_mask  # 裁剪标签，仅保留框内部分

red_label = np.zeros((*mask.shape, 4), dtype=np.uint8)
red_label[label == 1] = [0, 0, 255, 100]  # BGR+Alpha，红色半透明

# 将红色标签叠加到图像上
for c in range(3):
    img_with_box[:,:,c] = img_with_box[:,:,c] * (1 - red_label[:,:,3]/255) + red_label[:,:,c] * (red_label[:,:,3]/255)



# 绘制矩形框（红色，线宽=2）
cv2.rectangle(
    img_with_box,
    (y_min, x_min),  # 左上角坐标 (注意OpenCV是 (x,y) 顺序)
    (y_max, x_max),   # 右下角坐标
    color=(0, 0, 255),  # BGR格式的红色
    thickness=2
)
h, w = img.shape[:2]
img_with_box = img_with_box[50:h-50, 50:w-50]  # 上100，下50，左右各50
x_min_new = x_min - 50
x_max_new = x_max - 50
y_min_new = y_min - 50
y_max_new = y_max - 50

# 检查是否越界
x_min_new = max(0, x_min_new)
x_max_new = min(img_with_box.shape[0], x_max_new)
y_min_new = max(0, y_min_new)
y_max_new = min(img_with_box.shape[1], y_max_new)

# 提取 bounding box 内容
box_img = img_with_box[x_min_new+2:x_max_new-1, y_min_new+2:y_max_new-1]
# 保存结果
cv2.imwrite(r"G:\GliomaRecurrence\LinHongYin2210887.png", img_with_box, [cv2.IMWRITE_PNG_COMPRESSION, 0])
cv2.imwrite(r"G:\GliomaRecurrence\LinHongYin2210887_crop.png", box_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])