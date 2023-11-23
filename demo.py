import spacy
from word2number import w2n

# 加载英文模型
nlp = spacy.load("en_core_web_sm")
def pos_tag(sentence):
    nouns = []
    num_tag = {}
    targets = ['NOUN', 'PROPN']
    tokens = nlp(sentence)

    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.pos_ in targets:
            noun = token.text
            if i + 1 < len(tokens) and tokens[i+1].pos_ in targets:
                noun += ' ' + tokens[i+1].text
                i += 1
            num_tag[noun] = False

            if i > 0 and tokens[i-1].pos_ == 'NUM':
                count = w2n.word_to_num(tokens[i-1].text)
                nouns.extend([noun] * count)
                num_tag[noun] = True
            else:
                nouns.append(noun)
        i += 1
    return nouns, num_tag
# 示例句子
sentence = "buses and three cars and apple tree"

nouns, num_tag = pos_tag(sentence)
print(nouns)
print(num_tag)
