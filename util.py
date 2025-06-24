import os
import yaml
import numpy as np
from scipy.spatial.distance import cosine
from word2number import w2n
import spacy
nlp = spacy.load("en_core_web_sm")
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import re
from detectron2.data import MetadataCatalog
from sentence_transformers import SentenceTransformer
from tools.PITI.pretrained_diffusion.script_util import create_model_and_diffusion
from detectron2.utils.colormap import random_color
from tools.OpenSeeD.openseed.BaseModel import BaseModel
from tools.OpenSeeD.openseed import build_model
import torch
import pickle
import json
import blobfile as bf


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(list_image_files_recursively(full_path))
    return results

def load_config_dict_to_opt(opt, config_dict):
    if not isinstance(config_dict, dict):
        raise TypeError("Config must be a Python dictionary")
    for k, v in config_dict.items():
        k_parts = k.split('.')
        pointer = opt
        for k_part in k_parts[:-1]:
            if k_part not in pointer:
                pointer[k_part] = {}
            pointer = pointer[k_part]
            assert isinstance(pointer, dict), "Overriding key needs to be inside a Python dict."
        ori_value = pointer.get(k_parts[-1])
        pointer[k_parts[-1]] = v

def load_opt(conf_file):
    default_path = './tools/OpenSeeD/configs/openseed/openseed_swint_lang.yaml'
    if conf_file is  None:
        conf_file = default_path
    opt = {}
    with open(conf_file, encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    load_config_dict_to_opt(opt, config_dict)

    return opt

def uint82bin(n, count=8):
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r, g, b = 0, 0, 0
        id = i + 1
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            id = id >> 3
        cmap[i, 0] =  r
        cmap[i, 1] =  g
        cmap[i, 2] =  b
    return cmap

class Colorize(object):
    def __init__(self, n=182):
        self.cmap = labelcolormap(n)

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((3, size[0], size[1])) 
    
        for label in range(0, len(self.cmap)):
            mask = (label == gray_image ) 
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image

def category_to_coco(classifier, category, coco_categories, map_file):
    if category in map_file.keys():
        coco_category = map_file[category]

    else:
        categories = []
        for item in coco_categories:
            categories.append(item['category'])
        candi_labels = []
        output_labels = []
        for _ in range(3):
            for i in range(len(categories)):
                candi_labels.append(categories[i])
                if len(candi_labels) == 10 or i == len(categories) - 1:
                    # print(candi_labels)
                    output = classifier(category, candi_labels)
                    max_value = max(output['scores'])
                    max_index = output['scores'].index(max_value)
                    output_labels.append(output['labels'][max_index])
                    candi_labels = []
            categories = output_labels
            output_labels = []
        coco_category = categories[0]
        map_file[category] = coco_category

    coco_index = None
    for i, item in enumerate(coco_categories):
        if item['category'] == coco_category:
            coco_index = i
            break
    coco_color = coco_categories[coco_index]['color']
    return coco_category, coco_color, map_file


def get_nouns(sentence, tagger):
    nouns = []
    target = 'NOUN'
    adjective = 'ADJ'
    tokens = tagger(sentence).sentences[0].words

    i = 0
    while i < len(tokens):
        data = {
            'obj': '',
            'num': 1,
            'hasNum': False
        }
        token = tokens[i]
        if (token.pos == adjective and i + 1 < len(tokens) and tokens[i+1].pos == target) or token.pos == target:
            noun = [token.text]
            obj = []
            if token.pos == target:
                obj.append(token.text)

            j = i + 1
            while j < len(tokens) and tokens[j].pos == target:
                noun.append(tokens[j].text)
                obj.append(tokens[j].text)
                j += 1
            
            noun_pharse = ' '.join(noun)

            length = len(noun_pharse.split())
            if i > 0 and (tokens[i-1].pos == 'NUM' or tokens[i-1].text.lower() in ['a', 'an']):
                if tokens[i-1].pos == 'NUM':
                    count = w2n.word_to_num(tokens[i-1].text)
                else:
                    count = 1
                data['num'] = count
                data['hasNum'] = True
            obj_pharse = ' '.join(obj)
            data['obj'] = obj_pharse
            nouns.append(data)
            i = j - 1
        i += 1
    return nouns


def load_draw_model(options, model_path, device='cuda'):
    '''_summary_

    :param options: _description_
    :param model_path: _description_
    :param device: _description_, defaults to 'cuda'
    :return: _description_
    '''
    model_ckpt = torch.load(model_path, map_location='cpu')
    model, _ = create_model_and_diffusion(**options)
    model.load_state_dict(model_ckpt, strict=True)
    return model.eval().to(device)

def load_seg_model(model_path, conf_file=None, categories_path='./id-category-color.jsonl', mode='sem', device='cuda'):
    '''load segmentation model

    :param model_path: _description_
    :param conf_file: _description_, defaults to None
    :param categories_path: _description_, defaults to './id-category.jsonl'
    :param mode: _description_, defaults to 'sem'
    :param categories: _description_, defaults to {}
    :param device: _description_, defaults to 'cuda'
    :raises ValueError: _description_
    :return: _description_
    '''
    if mode not in ['sem', 'pano']:
        raise ValueError(f"Invalid mode: {mode}. Mode must be 'sem' or 'pano'.")
    opt = load_opt(conf_file)
    opt['WEIGHT'] = model_path
    model = BaseModel(opt, build_model(opt)).from_pretrained(opt['WEIGHT']).eval().to(device)
    if mode == 'sem':
        stuff_classes = []
        stuff_colors = []
        stuff_dataset_id_to_contiguous_id = {}

        with open(categories_path, 'r') as f:
            index = 0
            for line in f:
                data = json.loads(line)
                stuff_classes.append(data['category'])
                stuff_colors.append([data['id']] * 3)
                stuff_dataset_id_to_contiguous_id[index] = data['id']
                index += 1
        MetadataCatalog.get("semseg").set(
            stuff_colors=stuff_colors,
            stuff_classes=stuff_classes,
            stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
        )
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(stuff_classes, is_eval=True)
        metadata = MetadataCatalog.get('semseg')
        model.model.metadata = metadata
        model.model.sem_seg_head.num_classes = len(stuff_classes)
        return model, stuff_dataset_id_to_contiguous_id
    else:
        thing_classes = []
        stuff_classes = []

        with open(categories_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                if data['id'] < 90:
                    thing_classes.append(data['category'])
                else:
                    stuff_classes.append(data['category'])
        categories = {'thing': thing_classes, 'stuff': stuff_classes}
        thing_colors = [random_color(rgb=True, maximum=255).astype(np.int).tolist() for _ in range(len(thing_classes))]
        stuff_colors = [random_color(rgb=True, maximum=255).astype(np.int).tolist() for _ in range(len(stuff_classes))]
        thing_dataset_id_to_contiguous_id = {x:x for x in range(len(thing_classes))}
        stuff_dataset_id_to_contiguous_id = {x+len(thing_classes):x for x in range(len(stuff_classes))}

        MetadataCatalog.get("panoseg").set(
            thing_colors=thing_colors,
            thing_classes=thing_classes,
            thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
            stuff_colors=stuff_colors,
            stuff_classes=stuff_classes,
            stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
        )
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(thing_classes + stuff_classes, is_eval=False)
        metadata = MetadataCatalog.get('panoseg')
        model.model.metadata = metadata
        model.model.sem_seg_head.num_classes = len(thing_classes + stuff_classes)
        return model, categories

def load_embed_model(model_name_or_path, device='cuda'):
    model = SentenceTransformer(model_name_or_path).to(device)
    return model


