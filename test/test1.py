import numpy as np
from PIL import Image


image_path = '/home/wgy/multimodal/MuMo/image_bank/000000002592.jpg'
mask_path = '/home/wgy/multimodal/MuMo/mask_bank/000000002592.jpg'

image = Image.open(image_path)
mask = Image.open(mask_path)
black_pixel = np.array([0, 0, 0])
white_pixel = np.array([255, 255, 255])
target = [128, 64, 64]

new_image = np.array(image)
mask = np.array(mask)
inpaint_mask = np.zeros_like(mask)

for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
        if mask[i][j].tolist() == target:
            inpaint_mask[i][j] = white_pixel
            new_image[i][j] = white_pixel
        else:
            inpaint_mask[i][j] = black_pixel

inpaint_mask = Image.fromarray(inpaint_mask)
new_image = Image.fromarray(new_image)
inpaint_mask.save('./mask.png')
new_image.show('./new.png')