from openai import OpenAI
import json


client = OpenAI(api_key = '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')

sentence1 = "Decorated coffee cup and knife sitting on a patterned surface."
sentence2 = "there is a cup of ice cream and a knife on a table"

# prompt = f"""
# 	Given a list of categories and a specific word, determine which single category the word fits into. Output only the category name, and ensure it is one from the provided list. Do not include any additional information or explanations in the output.

#     Categories: {coco_categories}
#     Word: {target}

#     Expected Output:
# 	"""

prompt = f"""
    Given two sentences A and B, pay attention to the objects described in the two sentences and the number of objects. Ignore words that describe the characteristics of objects. Determine whether the objects and number described in sentence A are included in sentence B.
    Output only yes or no.

    Sentence A: {sentence1}
    Sentence B: {sentence2}

    Excepted Output:
"""

completion = client.completions.create(model='text-davinci-003', prompt=prompt)
print(completion.choices[0].text)


