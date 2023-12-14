import json

path1 = '/home/wgy/multimodal/MuMo/check_ablation.jsonl'
path2 = '/home/wgy/multimodal/MuMo/target_objs_ablation.jsonl'
path3 = '/home/wgy/multimodal/MuMo/check_ablation_filted.jsonl'

with open(path1, 'r') as f1, open(path2, 'r') as f2, open(path3, 'w') as f3:
    id2tar = {}
    id2gt = {}
    for line in f2:
        data =json.loads(line)
        id = data['name'].split('.')[0]
        tar_objs = data['tar_objs']
        gt = data['gt']
        id2tar[id] = tar_objs
        id2gt[id] = gt
    for line in f1:
        data = json.loads(line)
        flag = data['flag']
        if flag:
            id = data['imgid']
            candi_objs = data['candi_objs']
            tar_objs = id2tar[id.split('-')[0]]
            gt = id2gt[id.split('-')[0]]
            sample_data = {
                'imgid': id,
                'tar_objs': tar_objs,
                'candi_objs': candi_objs,
                'gt': gt
            }
            f3.write(json.dumps(sample_data) + '\n')