import os
import json
from transformers import pipeline
import torch
import copy
from tqdm import tqdm
import stanza
from word2number import w2n

pos_tagger = stanza.Pipeline(lang='en', processors='tokenize,pos')
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)
black_list = ['top']

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
        # with open(map_file, 'w') as f:
            # json.dump(map_file, f, indent=4)

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


def get_objs_from_caption(caption, coco_categories, map_file):
    with torch.no_grad():
        caption_objs = get_nouns(caption, pos_tagger)
        # ori_objs = copy.deepcopy(caption_objs)
        ori_objs = []
        captions = []
        for obj in caption_objs:
            if obj['obj'].lower() not in black_list:
                ori_objs.append(obj)
        for obj in ori_objs:
            obj['obj'], _, map_file = category_to_coco(classifier=classifier, category=obj['obj'].lower(), coco_categories=coco_categories, map_file=map_file)
            captions.append(obj['obj'])
        return captions, ori_objs, map_file


path1 = '/home/wgy/metaic_result/metaic_result/name_img_id_dict_metaic.jsonl'
path2 = '/home/wgy/metaic_result/po/all_info.jsonl'
save_path = '/home/wgy/metaic_result/po/result_labeled.jsonl'

coco_categories = []
with open('./id-category-color.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        coco_categories.append(data)
map_file_path = './category2coco.json'
with open(map_file_path, 'r') as f:
    map_file = json.load(f)

ics = ['vinvl', 'blip2', 'blip', 'git', 'ofa', 'vit_gpt2', 'azure']
with open(path1, 'r') as f1, open(path2, 'r') as f2, open(save_path, 'w') as f:
    for line1, line2 in tqdm(zip(f1, f2)):
        data1 = json.loads(line1)
        data2 = json.loads(line2)
        target = data1['inserted_obj']
        coco_target, _, _ = category_to_coco(classifier=classifier, category=target, coco_categories=coco_categories, map_file=map_file)
        for i in range(len(ics)):
            label_name = ics[i] + '_label'
            flag_name = ics[i] + '_flag'
            # cur_label = data2[label_name]
            cur_flag = data2[flag_name]
            if cur_flag == False:
                caption_name = ics[i] + 'd'
                caption = data2[caption_name]
                captions, _, map_file = get_objs_from_caption(caption, coco_categories, map_file)
                if coco_target not in captions:
                    data2[label_name] = [0]
                else:
                    data2[label_name] = [1]
            # for key in data2.keys():
            #     print(f'{key}: {data2[key]}')
        f.write(json.dumps(data2) + '\n')
print('finish')


