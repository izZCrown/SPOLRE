from PIL import Image
import os

img_dir = '/home/wgy/multimodal/MuMo/test/images'
list1 = os.listdir(img_dir)

save_path = '/home/wgy/multimodal/MuMo/test/numerr.png'

for item in list1:
    img_path = os.path.join(img_dir, item)
    print(img_path)
    image = Image.open(img_path)
    image.save(save_path)
    input1 = input('enter: ')