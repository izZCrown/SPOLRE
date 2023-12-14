from sklearn.metrics import cohen_kappa_score
import json

ics = ['vinvl', 'blip2', 'blip', 'git', 'ofa', 'vitgpt2', 'azure']

labels1 = []
labels2 = []

with open('/home/wgy/multimodal/result_final_labeled_filted.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        for i in range(len(ics)):
            label_name = ics[i] + '_label'
            cur_label = data[label_name]
            if cur_label != ["normal", "normal"]:
                labels1.append(cur_label[0])
                labels2.append(cur_label[1])



# Calculate Cohen's Kappa
kappa = cohen_kappa_score(labels1, labels2)
print(kappa)