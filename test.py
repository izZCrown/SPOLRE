import cv2
from PIL import Image
from simple_lama_inpainting import SimpleLama
lama = SimpleLama()
import numpy as np

def inpaint(image, mask):
    # mask = Image.fromarray(mask).convert('L')
    # exp_mask = np.zeros_like(mask)
    # kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # exp_mask = cv2.dilate(mask, kernel, iterations=1)
    # exp_mask = Image.fromarray(exp_mask).convert('L')
    inpaint_img = lama(image, exp_mask)
    # inpaint_img = lama(image, mask)
    return inpaint_img


ori_image_path = '//home/wgy/multimodal/MuMo/image_bank/000000010363.jpg'
ori_mask_path = '/home/wgy/multimodal/MuMo/mask_bank/000000010363/000000010363-0.png'

ori_image = Image.open(ori_image_path)
ori_image_np = np.array(ori_image)

ori_mask = Image.open(ori_mask_path)
ori_mask_np = np.array(ori_mask)

black_pixel = np.array([0, 0, 0])
white_pixel = np.array([255, 255, 255])


colors = [[128, 64, 0], [128, 128, 0]]
target_color = colors[0]

background = np.zeros_like(ori_image_np)

for i in range(ori_image_np.shape[0]):
    for j in range(ori_image_np.shape[1]):
        if ori_mask_np[i][j].tolist() != target_color and ori_mask_np[i][j].tolist() in colors:
            background[i][j] = white_pixel
        else:
            background[i][j] = black_pixel

# exp_mask = np.zeros_like(background)
kernel_size = 50
kernel = np.ones((kernel_size, kernel_size), np.uint8)
exp_mask = cv2.dilate(background, kernel, iterations=1)

for i in range(ori_image_np.shape[0]):
    for j in range(ori_image_np.shape[1]):
        if ori_mask_np[i][j].tolist() == target_color:
            exp_mask[i][j] = black_pixel
        # else:
        #     background[i][j] = black_pixel



exp_mask = Image.fromarray(exp_mask).convert('L')
inpaint_img = inpaint(ori_image, exp_mask)
# inpaint_img.save('/home/wgy/multimodal/MuMo/test/final.png')
inpaint_img.save('/home/wgy/multimodal/MuMo/test/mask.png')
