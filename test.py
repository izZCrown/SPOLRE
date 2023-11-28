from PIL import Image
import numpy as np
image = Image.open('/home/wgy/multimodal/MuMo/mask_bank/000000002592/000000002592-0.png')
image = np.array(image)
print(image.shape)