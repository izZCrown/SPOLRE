import json

path1 = '/home/wgy/multimodal/MuMo/check_1207.jsonl'
path2 = '/home/wgy/multimodal/caption_1207_5.jsonl'
path3 = '/home/wgy/multimodal/caption_1207_5_filter_false.jsonl'

with open(path1, 'r') as f1, open(path2, 'r') as f2, open(path3, 'w') as f3:
    for data1, data2 in zip(f1, f2):
        data1 = json.loads(data1)
        data2 = json.loads(data2)
        flag = data1['flag']
        if flag == False:
            f3.write(json.dumps(data2) + '\n')