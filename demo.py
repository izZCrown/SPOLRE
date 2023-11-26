from sentence_transformers import CrossEncoder
import json
import numpy as np
import torch
import torch.nn.functional as F

# model = CrossEncoder('cross-encoder/nli-deberta-v3-base')

# scores = []
# # score = model.predict([('man', 'dog')])[0]
# # print(score[1] - score[0])
# with open('/home/wgy/multimodal/MuMo/id-category-color.jsonl', 'r') as f:
#     for line in f:
#         data = json.loads(line)
#         category = data['category']
#         score = model.predict([('apple tree', category)])[0]
#         print(f'{category}: {score}')

# print(np.argmax(scores))
list1 = torch.tensor([-2.6020617, 1.8440553])
list2 = torch.tensor([-3.0912812, -0.7404097])
list1 = F.softmax(list1, dim=0)
list2 = F.softmax(list2, dim=0)
print(list1)
print(list2)
