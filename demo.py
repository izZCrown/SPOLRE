import json

with open('/home/wgy/multimodal/MuMo/image_cation.json', 'r') as f:
    data = json.load(f)

for key in data.keys():
    print(data[key])