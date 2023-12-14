import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
sys.path.append('./tools/PITI')
sys.path.append('./tools/OpenSeeD')
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline, pipeline
from util import get_nouns, category_to_coco, list_image_files_recursively
import torch
import copy
from tqdm import tqdm
import stanza
# pos_model_path = 'QCRI/bert-base-multilingual-cased-pos-english'
# pos_model = AutoModelForTokenClassification.from_pretrained(pos_model_path)
# pos_tokenizer = AutoTokenizer.from_pretrained(pos_model_path)
# pos_tagger = TokenClassificationPipeline(model=pos_model, tokenizer=pos_tokenizer)
pos_tagger = stanza.Pipeline(lang='en', processors='tokenize,pos')
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)
black_list = ['top']

def get_objs_from_caption(caption, coco_categories, map_file):
    with torch.no_grad():
        caption_objs = get_nouns(caption, pos_tagger)
        # ori_objs = copy.deepcopy(caption_objs)
        ori_objs = []
        for obj in caption_objs:
            if obj['obj'].lower() not in black_list:
                ori_objs.append(obj)
        for obj in ori_objs:
            obj['obj'], _, map_file = category_to_coco(classifier=classifier, category=obj['obj'].lower(), coco_categories=coco_categories, map_file=map_file)
        return caption_objs, ori_objs, map_file


path = '/home/wgy/multimodal/caption_1212_7.jsonl'
path1 = '/home/wgy/multimodal/MuMo/check_ablation_filted.jsonl'
save_path = '../all_info_ablation.jsonl'

coco_categories = []
with open('./id-category-color.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        coco_categories.append(data)

map_file_path = './category2coco.json'
with open(map_file_path, 'r') as f:
    map_file = json.load(f)

id2tarobj = {}
id2meltobj = {}
id2gt = {}

with open(path1, 'r') as f:
    for line in f:
        data = json.loads(line)
        id = data['imgid']
        tar_obj = []
        tar_objs = data['tar_objs']
        for item in tar_objs:
            sample_data = {
                'obj': item['obj'],
                'num': item['num']
            }
            tar_obj.append(sample_data)
        id2tarobj[id] = tar_obj
        id2meltobj[id] = data['candi_objs']
        id2gt[id] = data['gt']

with open(path, 'r') as f1, open(save_path, 'w') as f2:
    for line in tqdm(f1):
        data = json.loads(line)
        imgid = data['imgid']
        vinvl_caption = data['vinvl_caption']
        blip2_caption = data['blip2_caption']
        blip_caption = data['blip_caption']
        git_caption = data['git_caption']
        ofa_caption = data['ofa_caption']
        vitgpt2_caption = data['vitgpt2_caption']
        azure_caption = data['azure_caption']
        if imgid in id2gt.keys():
            tar_objs = id2tarobj[imgid]
            melt_objs = id2meltobj[imgid]
            gt = id2gt[imgid]
            try:
                vinvl_objs, vinvl_ori_objs, map_file = get_objs_from_caption(caption=vinvl_caption, coco_categories=coco_categories, map_file=map_file)
            except:
                vinvl_objs, vinvl_ori_objs = [], []
            try:
                blip2_objs, blip2_ori_objs, map_file = get_objs_from_caption(caption=blip2_caption, coco_categories=coco_categories, map_file=map_file)
            except:
                blip2_objs, blip2_ori_objs = [], []
            try:
                blip_objs, blip_ori_objs, map_file = get_objs_from_caption(caption=blip_caption, coco_categories=coco_categories, map_file=map_file)
            except:
                blip_objs, blip_ori_objs = [], []
            try:
                git_objs, git_ori_objs, map_file = get_objs_from_caption(caption=git_caption, coco_categories=coco_categories, map_file=map_file)
            except:
                git_objs, git_ori_objs = [], []
            try:
                ofa_objs, ofa_ori_objs, map_file = get_objs_from_caption(caption=ofa_caption, coco_categories=coco_categories, map_file=map_file)
            except:
                ofa_objs, ofa_ori_objs = [], []
            try:
                vitgpt2_objs, vitgpt2_ori_objs, map_file = get_objs_from_caption(caption=vitgpt2_caption, coco_categories=coco_categories, map_file=map_file)
            except:
                vitgpt2_objs, vitgpt2_ori_objs = [], []
            try:
                azure_objs, azure_ori_objs, map_file = get_objs_from_caption(caption=azure_caption, coco_categories=coco_categories, map_file=map_file)
            except:
                azure_objs, azure_ori_objs = [], []
            sample_data = {
                'id': imgid,
                'tar_objs': tar_objs,
                'melt_objs': melt_objs,
                'gt': gt,

                'vinvl_objs': vinvl_objs,
                'vinvl_ori_objs': vinvl_ori_objs,
                'vinvl': vinvl_caption,

                'blip2_objs': blip2_objs,
                'blip2_ori_objs': blip2_ori_objs,
                'blip2': blip2_caption,

                'blip_objs': blip_objs,
                'blip_ori_objs': blip_ori_objs,
                'blip': blip_caption,

                'git_objs': git_objs,
                'git_ori_objs': git_ori_objs,
                'git': git_caption,

                'ofa_objs': ofa_objs,
                'ofa_ori_objs': ofa_ori_objs,
                'ofa': ofa_caption,

                'vitgpt2_objs': vitgpt2_objs,
                'vitgpt2_ori_objs': vitgpt2_ori_objs,
                'vitgpt2': vitgpt2_caption,

                'azure_objs': azure_objs,
                'azure_ori_objs': azure_ori_objs,
                'azure': azure_caption,
            }
            f2.write(json.dumps(sample_data) + '\n')

with open(map_file_path, 'w') as f:
    json.dump(map_file, f, indent=4)
print('Finish...')

# directory = '/home/wgy/multimodal/gen_image_1202'

# all_files = list_image_files_recursively(directory)
# print(len(all_files))