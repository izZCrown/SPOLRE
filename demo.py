import spacy
from word2number import w2n

# 加载英文模型
nlp = spacy.load("en_core_web_sm")
def pos_tag(sentence):
    nouns = []
    targets = ['NOUN', 'PROPN']
    tokens = nlp(sentence)

    i = 0
    while i < len(tokens):
        data = {
            'obj': '',
            'num': 1,
            'hasNum': False
        }
        token = tokens[i]
        if token.pos_ in targets:
            noun = token.text
            if i + 1 < len(tokens) and tokens[i+1].pos_ in targets:
                noun += ' ' + tokens[i+1].text
                i += 1
            data['obj'] = noun

            length = len(noun.split())
            if i > 0 and (tokens[i-length].pos_ == 'NUM' or tokens[i-length].text.lower() in ['a', 'an']):
                if tokens[i-length].pos_ == 'NUM':
                    count = w2n.word_to_num(tokens[i-length].text)
                else:
                    count = 1
                data['num'] = count
                data['hasNum'] = True
            nouns.append(data)
        i += 1
    return nouns
# 示例句子
sentence = "a bus and three cars and three apple tree"

nouns = pos_tag(sentence)
print(nouns)
sentence = "bus and three cars and an apple tree"
nouns = pos_tag(sentence)
print(nouns)
