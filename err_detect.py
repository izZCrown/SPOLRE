import json
from tqdm import tqdm

caption_path = '/home/wgy/multimodal/all_info_1207.jsonl'
# caption_path = '/home/wgy/multimodal/all_info_false.jsonl'
save_path = '../result_6.jsonl'
# save_path = '../result_false.jsonl'
"""
keys:
id
tar_objs
ori_objs
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
ofa_objs
ofa_ori_objs
ofa
vitgpt2_objs
vitgpt2_ori_objs
vitgpt2
"""

def find_err(tar_objs, gen_objs):
    tar_dict = {}
    gen_dict = {}
    for gen_obj in gen_objs:
        gen_obj_name = gen_obj['obj']
        gen_obj_num = gen_obj['num']
        flag = gen_obj['hasNum']
        gen_dict[gen_obj_name] = [gen_obj_num, flag]
    for tar_obj in tar_objs:
        tar_obj_name = tar_obj['obj']
        tar_obj_num = tar_obj['num']
        tar_dict[tar_obj_name] = tar_obj_num
    # print(tar_dict)
    # print(gen_dict)
    error_list = []
    for key in tar_dict.keys():
        if key not in gen_dict.keys():
            if len(tar_dict.keys()) <= len(gen_dict.keys())-1:
                error_list.append("Misclassification")
            else:
                error_list.append("Omission")
    for key in tar_dict.keys():
        try:
            tar_num = tar_dict[key]
            gen_num = gen_dict[key][0]
            flag = gen_dict[key][1]
            if flag and tar_num != gen_num:
                error_list.append("NumErr")
        except:
            pass
    
    return error_list

with open(caption_path, 'r') as f1, open(save_path, 'w') as f2:
    for line in tqdm(f1):
        data = json.loads(line)
        imgid = data['id']
        tar_objs = data['tar_objs']
        vinvl_objs = data['vinvl_objs']
        # vinvl_err, vinvl_obj = find_err(tar_objs, vinvl_objs)
        vinvl_err = find_err(tar_objs, vinvl_objs)

        blip2_objs = data['blip2_objs']
        # blip2_err, blip2_obj = find_err(tar_objs, blip2_objs)
        blip2_err = find_err(tar_objs, blip2_objs)

        blip_objs = data['blip_objs']
        # blip_err, blip_obj = find_err(tar_objs, blip_objs)
        blip_err = find_err(tar_objs, blip_objs)

        git_objs = data['git_objs']
        # git_err, git_obj = find_err(tar_objs, git_objs)
        git_err = find_err(tar_objs, git_objs)

        ofa_objs = data['ofa_objs']
        # ofa_err, ofa_obj = find_err(tar_objs, ofa_objs)
        ofa_err = find_err(tar_objs, ofa_objs)

        vitgpt2_objs = data['vitgpt2_objs']
        # vitgpt2_err, vitgpt2_obj = find_err(tar_objs, vitgpt2_objs)
        vitgpt2_err = find_err(tar_objs, vitgpt2_objs)

        sample_data = {
            'imgid': imgid,
            'tar_objs': tar_objs,

            'vinvl_objs': vinvl_objs,
            'vinvl_err': vinvl_err,
            # 'vinvl_obj': vinvl_obj,

            'blip2_objs': blip2_objs,
            'blip2_err': blip2_err,
            # 'blip2_obj': blip2_obj,

            'blip_objs': blip_objs,
            'blip_err': blip_err,
            # 'blip_obj': blip_obj,

            'git_objs': git_objs,
            'git_err': git_err,
            # 'git_obj': git_obj,

            'ofa_objs': ofa_objs,
            'ofa_err': ofa_err,
            # 'ofa_obj': ofa_obj,

            'vitgpt2_objs': vitgpt2_objs,
            'vitgpt2_err': vitgpt2_err,
            # 'vitgpt2_obj': vitgpt2_obj,
        }
        f2.write(json.dumps(sample_data) + '\n')

# with open(save_path, 'r') as f:
#     total_err_num = 0
#     misclass_num = 0
#     omission_num = 0
#     err_num = 0
#     for line in f:
#         data = json.loads(line)
#         vinvl_err = data['vinvl_err']
#         blip2_err = data['blip2_err']
#         blip_err = data['blip_err']
#         ofa_err = data['ofa_err']
#         vitgpt2_err = data['vitgpt2_err']
#         err_list = [vinvl_err, blip2_err, blip_err, ofa_err, vitgpt2_err]
#         total_err_num += (5 - err_list.count("Normal"))
#         misclass_num += err_list.count("Misclassification")
#         omission_num += err_list.count("Omission")
#         err_num += err_list.count("NumErr")
# print(f"Total err: {total_err_num}")
# print(f"Misclassification: {misclass_num}")
# print(f"Omission: {omission_num}")
# print(f"Numerical Inaccuracy: {err_num}")
with open(save_path, 'r') as f:
    vinvl_err, blip2_err, blip_err, git_err, ofa_err, vitgpt2_err = [], [], [], [], [], []
    err_num = 0
    for line in f:
        data = json.loads(line)
        vinvl_err += data['vinvl_err']
        blip2_err += data['blip2_err']
        blip_err += data['blip_err']
        git_err += data['git_err']
        ofa_err += data['ofa_err']
        vitgpt2_err += data['vitgpt2_err']
    print(f"vinvl: Omission: {vinvl_err.count('Omission')}")
    print(f"vinvl: Misclassification: {vinvl_err.count('Misclassification')}")
    print(f"vinvl: NumErr: {vinvl_err.count('NumErr')}")

    print(f"blip2: Omission: {blip2_err.count('Omission')}")
    print(f"blip2: Misclassification: {blip2_err.count('Misclassification')}")
    print(f"blip2: NumErr: {blip2_err.count('NumErr')}")

    print(f"blip: Omission: {blip_err.count('Omission')}")
    print(f"blip: Misclassification: {blip_err.count('Misclassification')}")
    print(f"blip: NumErr: {blip_err.count('NumErr')}")

    print(f"git: Omission: {git_err.count('Omission')}")
    print(f"git: Misclassification: {git_err.count('Misclassification')}")
    print(f"git: NumErr: {git_err.count('NumErr')}")

    print(f"ofa: Omission: {ofa_err.count('Omission')}")
    print(f"ofa: Misclassification: {ofa_err.count('Misclassification')}")
    print(f"ofa: NumErr: {ofa_err.count('NumErr')}")

    print(f"vitgpt2: Omission: {vitgpt2_err.count('Omission')}")
    print(f"vitgpt2: Misclassification: {vitgpt2_err.count('Misclassification')}")
    print(f"vitgpt2: NumErr: {vitgpt2_err.count('NumErr')}")

