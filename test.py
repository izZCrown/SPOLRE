# import stanza
# from word2number import w2n
# import json
# nlp = stanza.Pipeline(lang='en', processors='tokenize,pos')  # 初始化英文NLP pipeline，包含分词和词性标注

# def get_nouns(sentence, tagger):
#     nouns = []
#     target = 'NOUN'
#     adjective = 'ADJ'
#     tokens = tagger(sentence).sentences[0].words

#     i = 0
#     while i < len(tokens):
#         data = {
#             'obj': '',
#             'num': 1,
#             'hasNum': False
#         }
#         token = tokens[i]
#         if (token.pos == adjective and i + 1 < len(tokens) and tokens[i+1].pos == target) or token.pos == target:
#             noun = [token.text]
#             obj = []
#             if token.pos == target:
#                 obj.append(token.text)

#             j = i + 1
#             while j < len(tokens) and tokens[j].pos == target:
#                 noun.append(tokens[j].text)
#                 j += 1
            
#             noun_pharse = ' '.join(noun)

#             length = len(noun_pharse.split())
#             print(tokens[i-1].text, tokens[i-1].pos)
#             if i > 0 and (tokens[i-1].pos == 'NUM' or tokens[i-1].text.lower() in ['a', 'an']):
#                 if tokens[i-1].pos == 'NUM':
#                     count = w2n.word_to_num(tokens[i-1].text)
#                 else:
#                     count = 1
#                 data['num'] = count
#                 data['hasNum'] = True
#             obj_pharse = ' '.join(obj)
#             data['obj'] = obj_pharse
#             nouns.append(data)
#             i = j - 1
#         i += 1
#     return nouns

# with open('/home/wgy/multimodal/image_caption_1201.json', 'r') as f:
#     data_list = json.load(f)

#     for key in data_list.keys():
#         # sentence = data_list[key]['caption']
#         sentence = 'A dog is reaching up to grab a Frisbee in his mouth.'
# # sentence = "there is a blacker table with two teddy bears bears and a book on it."
#         print(sentence)
#         print(get_nouns(sentence, nlp))

path = '/home/wgy/multimodal/gen_image_1205'

# list1 = os.listdir(path)
# print(len(list1))
import sys
sys.path.append('./tools/PITI')
sys.path.append('./tools/OpenSeeD')
import json
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline, pipeline
from util import get_nouns, category_to_coco, list_image_files_recursively
import torch
import copy
from tqdm import tqdm

print(len(list_image_files_recursively(path)))