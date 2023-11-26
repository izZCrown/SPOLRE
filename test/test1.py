import numpy as np
import cv2

# 创建一个示例数组，表示一个100x100的蓝色图像
width, height = 100, 100
blue_color = (255, 0, 0)  # BGR格式
layout = np.full((height, width, 3), blue_color, dtype=np.uint8)

# 保存图像
cv2.imwrite('blue_image.png', layout)
