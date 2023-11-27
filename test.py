import json

map_file = './category2coco.json'
with open(map_file, 'r') as f:
    coco_map = json.load(f)
print(coco_map)

category = 'cup'
print(category in coco_map.keys())