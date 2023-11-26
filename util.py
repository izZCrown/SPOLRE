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


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

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
        id = i + 1  # let's give 0 a color
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

def category_to_coco(model, category, coco_categories):
    '''Map objects to categories in Coco

    :param model: _description_
    :param category: _description_
    :param coco_categories: _description_
    :return: _description_
    '''
    category_embedding = model.encode([category.lower()])[0]
    distances = []
    for item in coco_categories:
        cos_dis = cosine(category_embedding, item['embed'])
        distances.append(cos_dis)

    min_dis = min(distances)
    min_index = distances.index(min_dis)
    category = coco_categories[min_index]['category']
    color = coco_categories[min_index]['color']
    return category, color

def token_from_nltk(sentence):
    words, tagged_tokens = [], []
    tokens = pos_tag(word_tokenize(sentence))
    for item in tokens:
        word = item[0]
        tag = item[1]
        if word[-1] == '.':
            word = word[:-1]
        if word != ' ' and word != '':
            words.append(word)
            tagged_tokens.append((word, tag))
    return words, tagged_tokens

def token_from_spacy(sentence):
    words, tagged_tokens = [], []
    tokens = nlp(sentence)
    for item in tokens:
        word = item.text
        tag = item.pos_
        if word[-1] == '.':
            word = word[:-1]
        if word != ' ' and word != '':
            words.append(word)
            tagged_tokens.append((word, tag))
    return words, tagged_tokens

def get_nouns(sentence):
    sentence = re.sub(r'[^\w\s,.]', ' ', sentence.rstrip()).replace('  ', ' ')
    if sentence[-1] == '.':
        sentence = sentence[:-1]
    nouns = []
    target_spacy = ['NOUN', 'PROPN']
    adjectives_spacy = ['ADJ']
    targets_nltk = ['NN', 'NNS', 'NNP', 'NNPS']
    adjectives_nltk = ['JJ', 'JJR', 'JJS']
    words_spacy, tokens_spacy = token_from_spacy(sentence)
    words_nltk, tokens_nltk = token_from_nltk(sentence)

        

    if words_spacy == words_nltk:
        tokens = []
        for token_spacy, token_nltk in zip(tokens_spacy, tokens_nltk):
            word = token_spacy[0]
            tag_spacy = token_spacy[1]
            tag_nltk = token_nltk[1]
            tag = tag_spacy
            if tag_spacy in target_spacy or tag_nltk in targets_nltk:
                tag = 'NN'
            elif tag_spacy in adjectives_spacy or tag_nltk in adjectives_nltk:
                tag = 'ADJ'
            elif tag_spacy == 'NUM' or tag_nltk == 'CD':
                tag = 'NUM'
            tokens.append((word, tag))
            
        i = 0
        while i < len(tokens):
            data = {
                'obj': '',
                'num': 1,
                'hasNum': False
            }
            token = tokens[i]
            if (token[1] == 'ADJ' and i + 1 < len(tokens) and tokens[i+1][1] == 'NN') or token[1] == 'NN':
                noun = [token[0]]

                j = i + 1
                while j < len(tokens) and tokens[j][1] == 'NN':
                    noun.append(tokens[j][0])
                    j += 1
                
                noun_pharse = ' '.join(noun)
                data['obj'] = noun[-1]

                length = len(noun_pharse.split())
                if i > 0 and (tokens[i-length][1] == 'NUM' or tokens[i-length][0].lower() in ['a', 'an']):
                    if tokens[i-length][1] == 'NUM':
                        count = w2n.word_to_num(tokens[i-length][0])
                    else:
                        count = 1
                    data['num'] = count
                    data['hasNum'] = True
                nouns.append(data)
                i = j - 1
            i += 1
    return nouns

