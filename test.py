import numpy as np
from PIL import Image

bk = '../back.png'
cup = '../cup.png'
knife = '../knife.png'

bk = np.array(Image.open(bk))
cup = np.array(Image.open(cup))
knife = np.array(Image.open(knife))

new_mask = np.zeros_like(bk)

for i in range(bk.shape[0]):
    for j in range(bk.shape[1]):
        if cup[i][j].tolist() != [255,255,255]:
            new_mask[i][j] = cup[i][j]
        elif knife[i][j].tolist() != [255,255,255]:
            new_mask[i][j] = knife[i][j]
        else:
            new_mask[i][j] = bk[i][j]

new_mask = Image.fromarray(new_mask)
new_mask.save('../stack.png')
