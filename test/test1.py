# from openai import OpenAI
import json
# import os
# key = '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'
# client = OpenAI(api_key=key)
# target = 'fire hydrant'
categories = []
with open('/home/wgy/multimodal/MuMo/id-category-color.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        categories.append(data['category'])
# prompt_categories = ', '.join(categories)

# # template = f"Here are some categories: {prompt_categories}, the word '{target}' should blong to category '<mask>'."
# # prompt = f"You are a taxonomist. Here are some categories: {prompt_categories}. What category do you think the word '{target}' should belong to? Just tell me the answer directly"
# # prompt = f"""
# # Here is a categories_list: {categories}.
# # What category do you think the word '{target}' should belong to?
# # You can only answer using elements from categories_list. 
# # Just tell me the answer directly.
# # """

# prompt = f"""
# Here are some nouns or noun phrases: {categories}.
# Which one do you think has the closest meaning to word '{target}'?
# Don't explain why. Just tell me directly what you choose is.
# """
# # print(prompt)

# completion = client.completions.create(
# 		model="gpt-3.5-turbo-instruct",
# 		prompt=prompt
# 	)
# print(completion.model_dump_json(indent=2))

from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)

target = 'red fire hydrant'

candi_labels = []
labels = []
for _ in range(3):
	print(categories)
	for i in range(len(categories)):
		candi_labels.append(categories[i])
		if len(candi_labels) == 10 or i == len(categories) - 1:
			# print(candi_labels)
			output = classifier(target, candi_labels)
			max_value = max(output['scores'])
			max_index = output['scores'].index(max_value)
			labels.append(output['labels'][max_index])
			candi_labels = []
	categories = labels
	labels = []
print(categories)
# print('--------------')
# print(labels)



