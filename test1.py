import json

# path1 = '/home/wgy/multimodal/MuMo/check_final.jsonl'
# path2 = '/home/wgy/multimodal/caption_1207_6.jsonl'
# path3 = '/home/wgy/multimodal/caption_1207_6_filter.jsonl'

# with open(path1, 'r') as f1, open(path2, 'r') as f2, open(path3, 'w') as f3:
#     for data1, data2 in zip(f1, f2):
#         data1 = json.loads(data1)
#         data2 = json.loads(data2)
#         flag = data1['flag']
#         if flag:
#             f3.write(json.dumps(data2) + '\n')
# path1 = '/home/wgy/multimodal/MuMo/check_final.jsonl'
path2 = '/home/wgy/multimodal/MuMo/check_final_filter.jsonl'

# img2tar = {}
# with open('/home/wgy/multimodal/MuMo/target_objs_final.jsonl', 'r') as f:
#     for line in f:
#         data = json.loads(line)
#         imgid = data['name'].split('.')[0]
#         tar_objs = data['tar_objs']
#         img2tar[imgid] = tar_objs

# with open(path1, 'r') as f1, open('/home/wgy/multimodal/MuMo/check_final2.jsonl', 'w') as f2:
#     for line in f1:
#         data = json.loads(line)
#         data['tar_objs'] = img2tar[data['imgid'].split('-')[0]]
#         f2.write(json.dumps(data) + '\n')


with open('/home/wgy/multimodal/MuMo/check_final.jsonl', 'r') as f1, open(path2, 'w') as f2:
    for line in f1:
        data = json.loads(line)
        flag = data['flag']
        if flag:
            imgid = data['imgid']
            tar_objs = data['tar_objs']
            candi_objs = data['candi_objs']
            gt = data['gt']
            sample_data = {
                'imgid': imgid,
                'tar_objs': tar_objs,
                'candi_objs': candi_objs,
                'gt': gt
            }
            f2.write(json.dumps(sample_data) + '\n')

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# import sys
# sys.path.append('./tools/PITI')
# sys.path.append('./tools/OpenSeeD')

# from argument import parser, print_args
# from einops import rearrange
# from PIL import Image, ImageChops
# import cv2
# import numpy as np
# np.random.seed(1)
# from util import mkdir, Colorize, category_to_coco, get_nouns, load_draw_model, load_seg_model, load_embed_model, list_image_files_recursively
# import torch
# from torchvision import transforms
# from tools.OpenSeeD.utils.arguments import load_opt_command
# from detectron2.data import MetadataCatalog
# from detectron2.utils.colormap import random_color
# from tools.OpenSeeD.openseed.BaseModel import BaseModel
# from tools.OpenSeeD.openseed import build_model
# from tools.OpenSeeD.utils.visualizer import Visualizer
# import json
# from sentence_transformers import SentenceTransformer
# import pickle
# from simple_lama_inpainting import SimpleLama
# lama = SimpleLama()

# from tools.PITI.pretrained_diffusion import dist_util, logger
# from torchvision.utils import make_grid
# from tools.PITI.pretrained_diffusion.script_util import create_model_and_diffusion
# from tools.PITI.pretrained_diffusion.image_datasets_mask import get_tensor
# from tools.PITI.pretrained_diffusion.train_util import TrainLoop
# from tools.PITI.pretrained_diffusion.glide_util import sample
# import time
# from tqdm import tqdm
# from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline
# from layout_editor import editor
# import traceback
# import copy
# import stanza

# classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)
# coco_categories = []
# with open('./id-category-color.jsonl', 'r') as f:
#     for line in f:
#         data = json.loads(line)
#         coco_categories.append(data)

# # category = "A white cat is laying by some tennis shoes"
# category = "living room"

# categories = []
# for item in coco_categories:
#     categories.append(item['category'])
# candi_labels = []
# output_labels = []
# for _ in range(3):
#     for i in range(len(categories)):
#         candi_labels.append(categories[i])
#         if len(candi_labels) == 10 or i == len(categories) - 1:
#             # print(candi_labels)
#             output = classifier(category, candi_labels)
#             max_value = max(output['scores'])
#             max_index = output['scores'].index(max_value)
#             output_labels.append(output['labels'][max_index])
#             candi_labels = []
#     categories = output_labels
#     output_labels = []
# coco_category = categories[0]
# print(coco_category)