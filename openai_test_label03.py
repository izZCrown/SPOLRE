from openai import OpenAI
import json
from tqdm import tqdm
import time

input_path = '/home/wgy/multimodal/result_final.jsonl'
output_path = '/home/wgy/multimodal/result_final_labeled03.jsonl'

client = OpenAI(api_key = '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')

def get_label(data, ic_name):
    label_name = ic_name + '_' + 'label'
    flag_name = ic_name + '_' + 'flag'
    label = data[flag_name]
    gt = data['gt']
    caption = data[ic_name]
    prompt = f"""
        Given two sentences A and B, pay attention to the objects described in the two sentences and the number of objects. Ignore words that describe the characteristics of objects. Determine whether the objects and number described in sentence A are included in sentence B.
        Output only yes or no.

        Sentence A: {gt}
        Sentence B: {caption}

        Excepted Output:
    """
    if label:
        manual_label = 'normal'
    else:
        manual_label = client.completions.create(model='text-davinci-003', prompt=prompt).choices[0].text
    data[label_name] = manual_label
    return data


with open(input_path, 'r') as f1, open(output_path, 'w') as f2:
    i = 0
    for line in tqdm(f1):
        if i % 5 == 3:
            # i += 1
            data = json.loads(line)
            while True:
                try:
                    for ic_name in ['vinvl', 'blip2', 'blip', 'git', 'ofa', 'vitgpt2', 'azure']:
                        data = get_label(data, ic_name)
                    break
                except Exception as e:
                    print('-----network error-----')
                    print(e)
                    print('-----------------------')
                    time.sleep(10)
            f2.write(json.dumps(data) + '\n')
        i += 1
print('finish')