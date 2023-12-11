import cv2
from PIL import Image
import os
from simple_lama_inpainting import SimpleLama
lama = SimpleLama()
import numpy as np

path1 = '/home/wgy/multimodal/images_1205/000000004495.jpg'
path2 = '/home/wgy/multimodal/mask_bank_1207/000000004495/000000004495-0.png'

image = Image.open(path1)
mask = Image.open(path2)

image_np = np.array(image)
mask_np = np.array(mask)

new_mask = np.zeros_like(image_np)

def mask_dilate(ori_mask, bw_mask, target_color, kernel_size=50):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    exp_bw_mask = cv2.dilate(bw_mask, kernel, iterations=1)
    
    if target_color != None:
        for i in range(ori_mask.shape[0]):
            for j in range(ori_mask.shape[1]):
                if ori_mask[i][j].tolist() == target_color:
                    exp_bw_mask[i][j] = np.array([0, 0, 0])
    return Image.fromarray(exp_bw_mask).convert('L')

for i in range(new_mask.shape[0]):
    for j in range(new_mask.shape[1]):
        if mask_np[i][j].tolist() == [64, 192, 192]:
            new_mask[i][j] = np.array([255, 255, 255])

new_mask = mask_dilate(mask_np, new_mask, [192, 192, 192])
# new_mask = Image.fromarray(new_mask)
new_mask.save('/home/wgy/multimodal/MuMo/test/melt_chair.png')
new_image = lama(image, new_mask)
new_image.save('/home/wgy/multimodal/MuMo/test/couch.png')

new_mask = np.zeros_like(image_np)

for i in range(new_mask.shape[0]):
    for j in range(new_mask.shape[1]):
        if mask_np[i][j].tolist() == [192, 192, 192]:
            new_mask[i][j] = np.array([255, 255, 255])

new_mask = mask_dilate(mask_np, new_mask, [64, 192, 192])
# new_mask = Image.fromarray(new_mask)
new_mask.save('/home/wgy/multimodal/MuMo/test/melt_couch.png')
new_image = lama(image, new_mask)
new_image.save('/home/wgy/multimodal/MuMo/test/chair.png')

new_mask = np.zeros_like(image_np)

for i in range(new_mask.shape[0]):
    for j in range(new_mask.shape[1]):
        if mask_np[i][j].tolist() == [192, 192, 192] or mask_np[i][j].tolist() == [64, 192, 192]:
            new_mask[i][j] = np.array([255, 255, 255])

new_mask = mask_dilate(mask_np, new_mask, None)
# new_mask = Image.fromarray(new_mask)
new_mask.save('/home/wgy/multimodal/MuMo/test/melt_all.png')
new_image = lama(image, new_mask)
new_image.save('/home/wgy/multimodal/MuMo/test/bk.png')


