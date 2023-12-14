import json
import os
from PIL import Image

# path1 = '/home/wgy/multimodal/result_final_labeled0.jsonl'
# path2 = '/home/wgy/multimodal/result_final_labeled1.jsonl'
path1 = '/home/wgy/metaic_result/no/all_info_labeled1.jsonl'
# path2 = '/home/wgy/rome_result/all_info_labeled2.jsonl'

save_path = '/home/wgy/metaic_result/po/result.jsonl'
save_path1 = '/home/wgy/metaic_result/po/result_labeled.jsonl'
# save_path1 = '/home/wgy/multimodal/result_final_labeled_filted.jsonl'
# save_path2 = '/home/wgy/multimodal/result_final_labeled_filted_temp.jsonl'
# dict_keys(['imgid', 'tar_objs', 'gt', 'vinvl_objs', 'vinvl_err', 'vinvl_flag', 'vinvl', 'blip2_objs', 'blip2_err', 'blip2_flag', 'blip2', 
#            'blip_objs', 'blip_err', 'blip_flag', 'blip', 'git_objs', 'git_err', 'git_flag', 'git', 'ofa_objs', 'ofa_err', 'ofa_flag', 'ofa', 
#            'vitgpt2_objs', 'vitgpt2_err', 'vitgpt2_flag', 'vitgpt2', 'azure_objs', 'azure_err', 'azure_flag', 'azure', 
#            'vinvl_label', 'blip2_label', 'blip_label', 'git_label', 'ofa_label', 'vitgpt2_label', 'azure_label'])


# 0表示确实是有问题，1表示没问题
def get_label(data1, target, err_num):
    label_name = target + '_label'
    label1 = data1[label_name].lower()
    if label1 == 'normal':
        pass
    # 两个描述是不是一个意思？auto认为有问题，手打标签如果认为是一个意思yes，则认为忽略了插入的物体，自动判断无误。
    # 如果标签是yes，说明两个句子一个意思，说明忽略了插入的物体，说明确实有问题，自动判断的没问题
    elif 'yes' in label1:
        # label1 = 1
        label1 = 0
    else:
    # 如果是no，说明两个句子不一个意思，说明正确识别了插入的物体，说明自动判断有问题
        # label1 = 0
        label1 = 1


    data1[label_name] = [label1]
    if data1[label_name] == [0]:
        err_num[0] += 1
    elif data1[label_name] == [1]:
        err_num[1] += 1
    return data1, err_num
# def get_label(data1, data2, target, err_num):
#     label_name = target + '_label'
#     label1 = data1[label_name].lower()
#     label2 = data2[label_name].lower()
#     if label1 == 'normal':
#         pass
#     elif 'yes' in label1:
#         # label1 = 1
#         label1 = 0
#     else:
#         # label1 = 0
#         label1 = 1

#     if label2 == 'normal':
#         pass
#     elif 'yes' in label2:
#         # label2 = 1
#         label2 = 0
#     else:
#         # label2 = 0
#         label2 = 1
#     data1[label_name] = [label1, label2]
#     if data1[label_name] == [0, 0]:
#         err_num[0] += 1
#     elif data1[label_name] == [1, 1]:
#         err_num[2] += 1
#     elif data1[label_name] == [0, 1] or data1[label_name] == [1, 0]:
#         err_num[1] += 1
#     return data1, err_num

# with open(path1, 'r') as f1, open(save_path, 'w') as f:
#     for line1 in f1:
#         data1 = json.loads(line1)
#         imgid1 = data1['imgid']

#         err_num = [0, 0]  # 不正常，一半正常，正常
#         for ic_name in ['vinvl', 'blip2', 'blip', 'git', 'ofa', 'vit_gpt2', 'azure']:
#             data1, err_num = get_label(data1, ic_name, err_num)
#         data1['err_num'] = err_num
#         f.write(json.dumps(data1) + '\n')
#     print('finish')
# with open(path1, 'r') as f1, open(path2, 'r') as f2, open(save_path, 'w') as f:
#     for line1, line2 in zip(f1, f2):
#         data1 = json.loads(line1)
#         data2 = json.loads(line2)
#         imgid1 = data1['imgid']
#         imgid2 = data2['imgid']

#         if imgid1 == imgid2:
#             err_num = [0, 0, 0]  # 不正常，一半正常，正常
#             for ic_name in ['vinvl', 'blip2', 'blip', 'git', 'ofa', 'vit_gpt2', 'azure']:
#                 data1, err_num = get_label(data1, data2, ic_name, err_num)
#             data1['err_num'] = err_num
#             f.write(json.dumps(data1) + '\n')
#     print('finish')

with open(save_path1, 'r') as f:
    manual = [0, 0, 0, 0, 0, 0, 0, 0]
    auto = [0, 0, 0, 0, 0, 0, 0, 0]
    ics = ['vinvl', 'blip2', 'blip', 'git', 'ofa', 'vit_gpt2', 'azure']
    for line in f:
        data = json.loads(line)
        for i in range(len(ics)):
            label_name = ics[i] + '_label'
            flag_name = ics[i] + '_flag'
            labels = data[label_name]
            if labels == [0]:
                manual[i] += 1
            if data[flag_name] == False:
                auto[i] += 1

count1 = 0
count2 = 0
for i in range(len(ics)):
    count1 += manual[i]
    count2 += auto[i]
    print(f"{ics[i]}: {round(manual[i]/auto[i], 2)}({manual[i]}/{auto[i]})")
print(count1)
print(count2)
print(count1/count2)


# with open(save_path, 'r') as f, open(save_path1, 'a+') as f1, open(save_path2, 'w') as f2:
# # with open(save_path, 'r') as f, open(save_path1, 'a+') as f1:
#     count = 0
#     for line in f:
#         data = json.loads(line)
#         num = data['err_num'][0]
#         if num > 0:
#             count += 1
#             f1.write(json.dumps(data) + '\n')
#         else:
#             f2.write(json.dumps(data) + '\n')

#     print(count)

# with open(save_path1, 'r') as f1, open(save_path2, 'w') as f2:
#     systems = ['vinvl', 'blip2', 'blip', 'git', 'ofa', 'vit_gpt2', 'azure']
#     for line in f1:
#         data = json.loads(line)
#         for i in range(len(systems)):
#             flag_name = systems[i] + '_flag'
#             label_name = systems[i] + '_label'
#             err_name = systems[i] + '_err'
#             errs = data[err_name]
#             if data[flag_name]:
#                 pass
#             else:
#                 if 'field' in data[systems[i]].lower() and 'Omission' in errs:
#                     errs.remove('Omission')
#                     data[err_name] = errs
#                     if len(errs) == 0:
#                         data[flag_name] = True
#                         data[label_name] = ["normal", "normal"]
#         f2.write(json.dumps(data) + '\n')


# with open(save_path2, 'r') as f:
#     systems = ['vinvl', 'blip2', 'blip', 'git', 'ofa', 'vit_gpt2', 'azure']
#     counts = [0, 0, 0, 0, 0, 0, 0]
#     counts_labeled = [0, 0, 0, 0, 0, 0, 0]
#     for line in f:
#         data = json.loads(line)
#         for i in range(len(systems)):
#             flag_name = systems[i] + '_flag'
#             label_name = systems[i] + '_label'
#             if data[flag_name]:
#                 pass
#             else:
#                 counts[i] += 1
#             if data[label_name] == [0, 0]:
#                 counts_labeled[i] += 1

# for i in range(len(systems)):
#     print(f'{systems[i]}: {round(counts_labeled[i]/counts[i], 2)}({counts_labeled[i]}/{counts[i]})')

# count = 0
# count_labeled = 0
# for i in range(len(systems)):
#     count += counts[i]
#     count_labeled += counts_labeled[i]

# print(count)
# print(count_labeled)
# print(round(count_labeled/count, 2))

# with open(save_path2, 'r') as f:
#     systems = ['vinvl', 'blip2', 'blip', 'git', 'ofa', 'vit_gpt2', 'azure']
#     err_type = ['omi', 'mis', 'num']
#     counts = [[], [], [], [], [], [], []]
#     print(counts)
#     for line in f:
#         data = json.loads(line)
#         for i in range(len(systems)):
#             err_name = systems[i] + '_err'
#             label_name = label_name = systems[i] + '_label'
#             if data[label_name] == [0, 0]:
#                 counts[i] += data[err_name]
# # print(counts)
# for i in range(len(counts)):
#     print(f"{systems[i]}: {err_type[0]}: {counts[i].count('Omission')}, {err_type[1]}: {counts[i].count('Misclassification')}, {err_type[2]}: {counts[i].count('NumErr')}, {counts[i].count('Omission')+counts[i].count('Misclassification')+counts[i].count('NumErr')}")


# with open(save_path1, 'r') as f1, open(save_path2, 'w') as f2:
#     systems = ['vinvl', 'blip2', 'blip', 'git', 'ofa', 'vitgpt2', 'azure']
#     for line in f1:
#         data = json.loads(line)
#         err_list = []
#         for i in range(len(systems)):
#             err_name = systems[i] + '_err'
#             err_list += data[err_name]
#         count = err_list.count('Omission')
#         if count != 0:
#             f2.write(json.dumps(data) + '\n')


# image_dir = '/home/wgy/multimodal/gen_image_final'
# with open(save_path2, 'r') as f:
#     for line in f:
#         data = json.loads(line)
#         imgid = data['imgid']
#         base_name = imgid.split('-')[0]
#         image_path = os.path.join(os.path.join(image_dir, base_name), imgid)
#         image = Image.open(image_path)
#         image.save('/home/wgy/multimodal/MuMo/test/numerr.png')
#         print(image_path)
#         input1 = input('press enter...')



