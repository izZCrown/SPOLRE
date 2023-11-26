import spacy
nlp = spacy.load("en_core_web_sm")
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from word2number import w2n
import json
import re

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

sentence = 'A living room with a couch and chair.'
print(get_nouns(sentence))



# sentence = "A black motorcycle parked at a building that says \"500\"."
# print(sentence)
# sentence = re.sub(r'[^\w\s,.]', ' ', sentence.rstrip()).replace('  ', ' ')
# # re.sub(r'[^\w\s,.]', '', sentence)
# print(sentence)
# print(token_from_nltk(sentence))
# print(token_from_spacy(sentence))

# with open('/home/wgy/multimodal/MuMo/captions_val2017.json', 'r') as f:
#     data = json.load(f)['annotations']
# for item in data:
#     sentence = item['caption']
#     sentence = re.sub(r'[^\w\s,.]', ' ', sentence.rstrip()).replace('  ', ' ')
#     if sentence[-1] == '.':
#         sentence = sentence[:-1]
#     nltk_list = token_from_nltk(sentence)
#     spacy_list = token_from_spacy(sentence)
#     if nltk_list != spacy_list:
#         print(nltk_list)
#         print(spacy_list)
#         print('-------------------')