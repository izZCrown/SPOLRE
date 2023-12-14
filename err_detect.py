import json
from tqdm import tqdm

caption_path = '/home/wgy/multimodal/all_info_final.jsonl'
# caption_path = '/home/wgy/multimodal/all_info_false.jsonl'
save_path = '../result_final_labeled_filted.jsonl'
# save_path = '../result_false.jsonl'
"""
keys:
id
tar_objs
melt_objs
gt
vinvl_objs
vinvl_ori_objs
vinvl
blip2_objs
blip2_ori_objs
blip2
blip_objs
blip_ori_objs
blip
git_objs
git_ori_objs
git
ofa_objs
ofa_ori_objs
ofa
vitgpt2_objs
vitgpt2_ori_objs
vitgpt2
azure_objs
azure_ori_objs
azure
"""

# def find_err(tar_objs, gen_objs, melt_objs):
#     tar_dict = {}
#     gen_dict = {}
#     for gen_obj in gen_objs:
#         gen_obj_name = gen_obj['obj']
#         gen_obj_num = gen_obj['num']
#         flag = gen_obj['hasNum']
#         if gen_obj_name not in melt_objs:
#             gen_dict[gen_obj_name] = [gen_obj_num, flag]
#     for tar_obj in tar_objs:
#         tar_obj_name = tar_obj['obj']
#         tar_obj_num = tar_obj['num']
#         tar_dict[tar_obj_name] = tar_obj_num

#     error_list = []
#     for key in tar_dict.keys():
#         if key not in gen_dict.keys():
#             if len(tar_dict.keys()) <= len(gen_dict.keys())-1:
#                 error_list.append("Misclassification")
#             else:
#                 error_list.append("Omission")
#     for key in tar_dict.keys():
#         try:
#             tar_num = tar_dict[key]
#             gen_num = gen_dict[key][0]
#             flag = gen_dict[key][1]
#             if flag and tar_num != gen_num:
#                 error_list.append("NumErr")
#         except:
#             pass
    
#     return error_list

# with open(caption_path, 'r') as f1, open(save_path, 'w') as f2:
#     for line in f1:
#         data = json.loads(line)
#         imgid = data['id']
#         tar_objs = data['tar_objs']
#         melt_objs = data['melt_objs']
#         gt = data['gt']
#         vinvl_objs = data['vinvl_objs']
#         # vinvl_err, vinvl_obj = find_err(tar_objs, vinvl_objs)
#         vinvl_err = find_err(tar_objs, vinvl_objs, melt_objs)

#         blip2_objs = data['blip2_objs']
#         # blip2_err, blip2_obj = find_err(tar_objs, blip2_objs)
#         blip2_err = find_err(tar_objs, blip2_objs, melt_objs)

#         blip_objs = data['blip_objs']
#         # blip_err, blip_obj = find_err(tar_objs, blip_objs)
#         blip_err = find_err(tar_objs, blip_objs, melt_objs)

#         git_objs = data['git_objs']
#         # git_err, git_obj = find_err(tar_objs, git_objs)
#         git_err = find_err(tar_objs, git_objs, melt_objs)

#         ofa_objs = data['ofa_objs']
#         # ofa_err, ofa_obj = find_err(tar_objs, ofa_objs)
#         ofa_err = find_err(tar_objs, ofa_objs, melt_objs)

#         vitgpt2_objs = data['vitgpt2_objs']
#         # vitgpt2_err, vitgpt2_obj = find_err(tar_objs, vitgpt2_objs)
#         vitgpt2_err = find_err(tar_objs, vitgpt2_objs, melt_objs)

#         azure_objs = data['azure_objs']
#         azure_err = find_err(tar_objs, azure_objs, melt_objs)

#         sample_data = {
#             'imgid': imgid,
#             'tar_objs': tar_objs,
#             'gt': gt,

#             'vinvl_objs': vinvl_objs,
#             'vinvl_err': vinvl_err,
#             'vinvl_flag': (len(vinvl_err) == 0),
#             'vinvl': data['vinvl'],
#             # 'vinvl_obj': vinvl_obj,

#             'blip2_objs': blip2_objs,
#             'blip2_err': blip2_err,
#             'blip2_flag': (len(blip2_err) == 0),
#             'blip2': data['blip2'],
#             # 'blip2_obj': blip2_obj,

#             'blip_objs': blip_objs,
#             'blip_err': blip_err,
#             'blip_flag': (len(blip_err) == 0),
#             'blip': data['blip'],
#             # 'blip_obj': blip_obj,

#             'git_objs': git_objs,
#             'git_err': git_err,
#             'git_flag': (len(git_err) == 0),
#             'git': data['git'],
#             # 'git_obj': git_obj,

#             'ofa_objs': ofa_objs,
#             'ofa_err': ofa_err,
#             'ofa_flag': (len(ofa_err) == 0),
#             'ofa': data['ofa'],
#             # 'ofa_obj': ofa_obj,

#             'vitgpt2_objs': vitgpt2_objs,
#             'vitgpt2_err': vitgpt2_err,
#             'vitgpt2_flag': (len(vitgpt2_err) == 0),
#             'vitgpt2': data['vitgpt2'],
#             # 'vitgpt2_obj': vitgpt2_obj,

#             'azure_objs': azure_objs,
#             'azure_err': azure_err,
#             'azure_flag': (len(azure_err) == 0),
#             'azure': data['azure'],
#         }
#         f2.write(json.dumps(sample_data) + '\n')

with open(save_path, 'r') as f:
    total_err_num = 0
    misclass_num = 0
    omission_num = 0
    err_num = [0, 0, 0, 0, 0, 0, 0]
    ic_list = ['vinvl_err', 'blip2_err', 'blip_err', 'git_err', 'ofa_err', 'vitgpt2_err', 'azure_err']
    for line in f:
        data = json.loads(line)
        vinvl_err = data['vinvl_err']
        blip2_err = data['blip2_err']
        blip_err = data['blip_err']
        git_err = data['git_err']
        ofa_err = data['ofa_err']
        vitgpt2_err = data['vitgpt2_err']
        azure_err = data['azure_err']
        err_list = [vinvl_err, blip2_err, blip_err, git_err, ofa_err, vitgpt2_err, azure_err]
        for i in range(len(err_list)):
            if len(err_list[i]) != 0:
                err_num[i] = err_num[i] + 1
for i in range(len(err_num)):
    print(f'{ic_list[i]}: {err_num[i]}')
    total_err_num += err_num[i]
print(f'total err: {total_err_num}')
print('-----------------------')
# print(f"Total err: {total_err_num}")
# print(f"Misclassification: {misclass_num}")
# print(f"Omission: {omission_num}")
# print(f"Numerical Inaccuracy: {err_num}")
with open(save_path, 'r') as f:
    vinvl_err, blip2_err, blip_err, git_err, ofa_err, vitgpt2_err, azure_err = [], [], [], [], [], [], []
    for line in f:
        data = json.loads(line)
        vinvl_err += data['vinvl_err']
        blip2_err += data['blip2_err']
        blip_err += data['blip_err']
        git_err += data['git_err']
        ofa_err += data['ofa_err']
        vitgpt2_err += data['vitgpt2_err']
        azure_err += data['azure_err']
    print(f"vinvl: Omission: {vinvl_err.count('Omission')}")
    print(f"vinvl: Misclassification: {vinvl_err.count('Misclassification')}")
    print(f"vinvl: NumErr: {vinvl_err.count('NumErr')}")
    print('-----------------------')

    print(f"blip2: Omission: {blip2_err.count('Omission')}")
    print(f"blip2: Misclassification: {blip2_err.count('Misclassification')}")
    print(f"blip2: NumErr: {blip2_err.count('NumErr')}")
    print('-----------------------')

    print(f"blip: Omission: {blip_err.count('Omission')}")
    print(f"blip: Misclassification: {blip_err.count('Misclassification')}")
    print(f"blip: NumErr: {blip_err.count('NumErr')}")
    print('-----------------------')

    print(f"git: Omission: {git_err.count('Omission')}")
    print(f"git: Misclassification: {git_err.count('Misclassification')}")
    print(f"git: NumErr: {git_err.count('NumErr')}")
    print('-----------------------')

    print(f"ofa: Omission: {ofa_err.count('Omission')}")
    print(f"ofa: Misclassification: {ofa_err.count('Misclassification')}")
    print(f"ofa: NumErr: {ofa_err.count('NumErr')}")
    print('-----------------------')

    print(f"vitgpt2: Omission: {vitgpt2_err.count('Omission')}")
    print(f"vitgpt2: Misclassification: {vitgpt2_err.count('Misclassification')}")
    print(f"vitgpt2: NumErr: {vitgpt2_err.count('NumErr')}")
    print('-----------------------')

    print(f"azure: Omission: {azure_err.count('Omission')}")
    print(f"azure: Misclassification: {azure_err.count('Misclassification')}")
    print(f"azure: NumErr: {azure_err.count('NumErr')}")
    print('-----------------------')

