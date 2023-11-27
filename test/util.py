import random
import cv2
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import pickle as pk
import copy

RAW = 0     # 原mask图
TRANSLATE = 1   # 平移
REVOLVE = 2     # 旋转
SCALING = 3     # 放缩
MIRROR = 4      # 镜像
# SCALING_TRANSLATE = 5 # 平移+旋转
CHANGE_TYPE_NUM = 4 # 可选择的变换数量
BACKGROUND = (255, 255, 255) # 背景色
# multiple_list = [0.4, 0.5, 0.6, 0.7, 0.8, 1.2, 1.3, 1.4, 1.5]  # 放缩倍数
multiple_list = [0.7, 0.8, 0.9, 1.1, 1.2, 1.3]  # 放缩倍数


# 平移
# src_img = cv2.imread(path)
# obj_info = {'rgb': [*, *, *], 'x_max': *, 'x_min': *, 'y_max': *, 'y_min': *}
def translate(src_img, obj_info):
    rows, cols, chnl = src_img.shape
    # delta_x>0右移，delta_x<0左移
    # delta_y>0下移，delta_y<0上移
    # 左上角为原点 右为x轴正向 下为y轴正向
    x_left = -obj_info['x_min']
    x_right = cols - obj_info['x_max']
    y_up = -obj_info['y_min']
    y_down = rows - obj_info['y_max']
    x_mov = int(random.uniform(x_right, x_left))
    y_mov = int(random.uniform(y_down, y_up))

    # 平移矩阵 M = np.float32([[1, 0, x], [0, 1, y]]) 水平方向移动x 竖直方向移动y
    M = np.float32([[1, 0, x_mov], [0, 1, y_mov]])
    # cv2.warpAffine(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) → dst
    # src - 输入图像。
    # M - 变换矩阵。
    # dsize - 输出图像的大小。
    # flags - 插值方法的组合（int类型！）
    # borderMode - 边界像素模式（int类型！）
    # borderValue - （重点！）边界填充值; 默认情况下，它为0
    dst_img = cv2.warpAffine(src_img, M, (cols, rows), borderValue=BACKGROUND)

    # 查看移动是否合理
    # watch(src_img, dst_img)

    return dst_img  # ndarray类型

# angle : 旋转角度
def revolve(src_img, obj_info, angle):
    rows, cols, chnl = src_img.shape
    x_mid = (obj_info['x_max'] + obj_info['x_min']) / 2
    y_mid = (obj_info['y_max'] + obj_info['y_min']) / 2
    # 旋转参数：旋转中心，旋转角度，旋转后的缩放比例
    M = cv2.getRotationMatrix2D((x_mid, y_mid), angle, 1)
    dst_img = cv2.warpAffine(src_img, M, (cols, rows), borderValue=BACKGROUND)
    # 查看移动是否合理
    # watch(src_img, dst_img)
    return dst_img

# multiple : 缩放比例
def scaling(src_img, obj_info, multiple):
    rows, cols, chnl = src_img.shape
    x_mid = (obj_info['x_max'] + obj_info['x_min']) / 2
    y_mid = (obj_info['y_max'] + obj_info['y_min']) / 2
    # 旋转参数：旋转中心，旋转角度，旋转后的缩放比例
    M = cv2.getRotationMatrix2D((x_mid, y_mid), 0, multiple)
    dst_img = cv2.warpAffine(src_img, M, (cols, rows), borderValue=BACKGROUND)
    # 查看移动是否合理
    # watch(src_img, dst_img)
    return dst_img

# direction : 0以X轴对称翻转，>0以Y轴对称翻转，<0以X轴Y轴同时翻转
def mirror(src_img, direction):
    dst_img = cv2.flip(src_img, direction)
    # 查看移动是否合理
    # watch(src_img, dst_img)
    return dst_img

# 将img_b叠加到img_a上 重合区域以img_b覆盖img_a
def merge(img_a, img_b):
    width = img_a.shape[0]
    height = img_b.shape[1]
    for i in range(width):
        for j in range(height):
            if img_b[i][j][0] != BACKGROUND[0] and img_b[i][j][1] != BACKGROUND[1] and img_b[i][j][2] != BACKGROUND[2]:
                img_a[i][j] = img_b[i][j]
    return img_a

# 将imgs:list中的图片全部叠加在一起
def img_merge(img_bg, imgs):
    # img_bg = np.zeros((rows, cols, 3), np.uint8) # 背景图
    # img_bg.fill(BACKGROUND[0])
    for index in imgs:
        img_bg = merge(img_bg, index["mask"])
    return img_bg

# 输入对象mask图 返回对应的info
def read_obj_info(obj_mask):
    obj_info = {}
    point_x_set = set()
    point_y_set = set()
    size = 0
    color = obj_mask[0][0]
    for j in range(0, obj_mask.shape[0]):
        for i in range(0, obj_mask.shape[1]):
            if obj_mask[j][i][0] != BACKGROUND[0] or obj_mask[j][i][1] != BACKGROUND[1] or obj_mask[j][i][2] != BACKGROUND[2]:
                point_x_set.add(i)
                point_y_set.add(j)
                color = obj_mask[j][i]
                size += 1
    obj_info["x_max"] = max(point_x_set)
    obj_info["x_min"] = min(point_x_set)
    obj_info["y_max"] = max(point_y_set)
    obj_info["y_min"] = min(point_y_set)
    obj_info["size"] = size
    obj_info["color"] = color.tolist()
    return obj_info

# 保存layout图片
def save_layout(image_id: int, mask):
    global now_idx

    # TODO 保存layout图片的名称在此处修改
    now_idx += 1
    mask_save_path = dst_dir + str(now_idx) + ".png"  # 保存原mask图
    idx_imgid_json[now_idx] = image_id

    cv2.imwrite(mask_save_path, mask)
    print(mask_save_path)

def get_background(image_id:int):
    # TODO 读取背景图片的路径可以在此处修改
    return cv2.imread(bg_dir+str(image_id)+".png")

# 读取image_id图片的所有对象mask图和相应的信息
# def get_obj_img(image_id:int):
#     obj_list = []
#     # TODO 读取图片对象的路径可以在此处修改
#     target_path = obj_dir+str(image_id)+"/"
#     all_obj = os.listdir(target_path)
#     for obj in all_obj:
#         # TODO 对象类别名读取
#         obj_name = obj.split(".")[0].rsplit("_", 1)[0]
#         obj_mask = cv2.imread(target_path+obj)
#         obj_info= read_obj_info(obj_mask)
#         # 对象类别 对象mask 对象信息
#         obj_list.append({"cat":obj_name, "mask":obj_mask, "info":obj_info})
#     return obj_list

def get_obj_img(image_dir, image_list):
    obj_list = []
    for image_name in image_list:
        image_path = os.path.join(image_dir, image_name)
        category = os.path.splitext(image_name)[0]
        mask = cv2.imread(image_path)
        info = read_obj_info(mask)
        data = {
            'category': category,
            'mask': mask,
            'info': info
        }
        obj_list.append(data)
    return obj_list


# 对mask图片进行变换
def obj_mask_change(obj):
    src_mask = obj["mask"]
    src_info = obj["info"]
    dst_dict = {}
    dst_dict["category"] = obj["category"]
    print("change cat:", obj["category"])

    # choice = random.randint(1, CHANGE_TYPE_NUM)
    choice = int(random.uniform(1, CHANGE_TYPE_NUM+1))
    if choice == TRANSLATE:
        print("平移(TRANSLATE)")
        dst_dict["mask"] = translate(src_mask, src_info)

    elif choice == REVOLVE:
        print("旋转(REVOLVE)")
        # angle = int(random.uniform(0 + 1, 360 - 1))
        angle = int(random.uniform(1, 90))
        dst_dict["mask"] = revolve(src_mask, src_info, angle)

    elif choice == SCALING:
        print("放缩(SCALING)")
        multiple = multiple_list[int(random.uniform(0, len(multiple_list)))]
        dst_dict["mask"] = scaling(src_mask, src_info, multiple)

    elif choice == MIRROR:
        print("镜像(MIRROR)")
        direction = int(random.uniform(-3, 3))
        dst_dict["mask"] = mirror(src_mask, direction)

    dst_dict["info"] = read_obj_info(dst_dict["mask"])
    return dst_dict

def sortBySize(elem):
    return elem["info"]["size"]

# 生成一张新的layout图
def single_layout_generate(index, output_path, bg, obj_list, obj_num, change_times):
    # 变换次数
    for time in range(change_times):
        # obj_idx = random.randint(0, obj_num-1)
        obj_idx = int(random.uniform(0, obj_num))
        obj_list[obj_idx] = obj_mask_change(obj_list[obj_idx])

    obj_list.sort(key=sortBySize, reverse=True)
    layout = img_merge(bg, obj_list)
    # 然后保存layout
    # save_layout(int(imgid), layout)
    save_name = f'{index}.png'
    save_path = os.path.join(output_path, save_name)
    cv2.imwrite(save_path, layout.astype(np.uint8))

def edit(image_dir, output_path, step, gen_num):
    image_dir = '/home/wgy/multimodal/MuMo/output/000000002592'
    for index in tqdm(range(gen_num)):
        all_img = os.listdir(image_dir)
        bg_img = 'background.png'
        all_img.remove(bg_img)
        bg_img_path = os.path.join(image_dir, bg_img)
        background = cv2.imread(bg_img_path)
        obj_list = get_obj_img(image_dir=image_dir, image_list=all_img)
        single_layout_generate(index, output_path, background, obj_list, len(obj_list), step)

image_dir = '/home/wgy/multimodal/MuMo/output/000000002592'
output_path = '/home/wgy/multimodal/MuMo/gen_mask/000000002592'

edit(image_dir, output_path, 10, 10)

    


