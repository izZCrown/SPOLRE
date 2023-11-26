from simple_lama_inpainting import SimpleLama
from PIL import Image

simple_lama = SimpleLama()

img_path = "/home/wgy/multimodal/MuMo/output/000000090208.jpg"
mask_path = "/home/wgy/multimodal/MuMo/mask_bank/test-1.png"

image = Image.open(img_path)
mask = Image.open(mask_path).convert('L')

result = simple_lama(image, mask)
result.save("/home/wgy/multimodal/MuMo/mask_bank/inpaint.png")

# from PIL import Image, ImageChops
# import numpy as np
# import cv2

# # 加载遮罩图像
# mask_path = '/home/wgy/multimodal/MuMo/mask_bank/test.png'
# mask = Image.open(mask_path)

# # 将PIL图像转换为NumPy数组
# mask_np = np.array(mask)

# # 创建一个与原遮罩相同大小的数组，初始化为黑色
# expanded_mask_np = np.zeros_like(mask_np)

# # 设置膨胀的结构元素的大小
# kernel_size = 20
# kernel = np.ones((kernel_size, kernel_size), np.uint8)

# # 使用OpenCV进行膨胀操作扩大白色区域
# expanded_mask_np = cv2.dilate(mask_np, kernel, iterations=1)

# # 将处理后的NumPy数组转换回PIL图像
# expanded_mask = Image.fromarray(expanded_mask_np)

# # 保存结果
# output_path = '/home/wgy/multimodal/MuMo/mask_bank/test-1.png'
# expanded_mask.save(output_path)

# # 返回保存后的文件路径
# output_path
