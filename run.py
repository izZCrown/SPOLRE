import os
import sys
sys.path.append('./tools/PITI')
sys.path.append('./tools/OpenSeeD')

from einops import rearrange
from PIL import Image
import cv2
import numpy as np
np.random.seed(1)
from util import mkdir, Colorize, category_to_coco, get_nouns, load_draw_model, load_seg_model
import torch
from torchvision import transforms
import json
from simple_lama_inpainting import SimpleLama
lama = SimpleLama()
from tools.PITI.pretrained_diffusion.image_datasets_mask import get_tensor
from tools.PITI.pretrained_diffusion.glide_util import sample
import time
from tqdm import tqdm
from transformers import pipeline
from layout_editor import editor
import traceback
import copy
import stanza
import argparse

BLACKPIXEL = np.array([0, 0, 0])
WHITEPIXEL = np.array([255, 255, 255])

# ==========load model==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open('./PITI_model_options.json', 'r') as f:
    options = json.load(f)
print('loading drawing model...')
base_model_path = '/home/wgy/multimodal/tools/ckpt/base_mask.pt'
base_model = load_draw_model(options['base_model'], base_model_path, device=device)

sample_model_path = '/home/wgy/multimodal/tools/ckpt/upsample_mask.pt'
sample_model = load_draw_model(options['sample_model'], sample_model_path, device=device)

seg_model_path = '/home/wgy/multimodal/tools/ckpt/model_state_dict_swint_51.2ap.pt'
t = []
t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
transform = transforms.Compose(t)

categories_path = './id-category-color.jsonl'
print('loading semseg model...')
semseg_model, color_map = load_seg_model(model_path=seg_model_path, categories_path=categories_path, mode='sem', device=device)
colorizer = Colorize(182)

print('loading panoseg model...')
panoseg_model, pano_categories = load_seg_model(model_path=seg_model_path, mode='pano', device=device)

print('loading classifier...')
if device == 'cuda':
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)
else:
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

print('loading pos tagger...')

pos_tagger = stanza.Pipeline(lang='en', processors='tokenize,pos')
print('loading models completed')
# ==============================
black_list = ['top']

def parse_args(input_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_path",
        type=str,
        default="./images",
        required=True,
        help="The path of seed images",
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default="./masks",
        required=True,
        help="Save path of the image masks",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./outputs",
        required=True,
        help="Save path of the generated images",
    )
    parser.add_argument(
        "--obj_mask_path",
        type=str,
        default="./obj_masks",
        required=True,
        help="Save path of object masks",
    )
    parser.add_argument(
        "--caption_path",
        type=str,
        default="./captions.jsonl",
        required=True,
        help="Path to the captions",
    )
    parser.add_argument(
        "--target_obj_path",
        type=str,
        default="./target_objs.jsonl",
        required=True,
        help="Save path of target object information",
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


def re_draw(image_path, output_path, num_samples=1, sample_c=1.3, sample_step=100, device='cuda'):
    '''Use PITI to draw according to the given mask

    :param image_path: mask path
    :param output_path: save path for generating images, defaults to './image_bank/'
    :param num_samples: number of generating images, defaults to 1
    :param sample_c: _description_, defaults to 1.3
    :param sample_step: _description_, defaults to 100
    :param device: _description_, defaults to 'cuda'
    '''
    mkdir(output_path)

    image = Image.open(image_path)
    label = image.convert("RGB").resize((256, 256), Image.NEAREST)
    image = np.array(image)
    label_tensor = get_tensor()(label)

    model_kwargs = {"ref":label_tensor.unsqueeze(0).repeat(num_samples, 1, 1, 1)}

    with torch.no_grad():
        samples_lr =sample(
            glide_model=base_model,
            glide_options=options['base_model'],
            side_x=64,
            side_y=64,
            prompt=model_kwargs,
            batch_size=num_samples,
            guidance_scale=sample_c,
            device=device,
            prediction_respacing=str(sample_step),
            upsample_enabled=False,
            upsample_temp=0.997,
            mode='coco',
        )

        samples_lr = samples_lr.clamp(-1, 1)

        tmp = (127.5 * (samples_lr + 1.0)).int() 
        model_kwargs['low_res'] = tmp / 127.5 - 1.

        samples_hr =sample(
            glide_model=sample_model,
            glide_options=options['sample_model'],
            side_x=256,
            side_y=256,
            prompt=model_kwargs,
            batch_size=num_samples,
            guidance_scale=1,
            device=device,
            prediction_respacing="fast27",
            upsample_enabled=True,
            upsample_temp=0.997,
            mode='coco',
        )

        ori_name = os.path.basename(image_path)
        index = 0
        for hr in samples_hr:
            save_name = os.path.splitext(ori_name)[0] + '-' + str(index) + '.png'
            save_path = os.path.join(output_path, save_name)
            hr_img = 255. * rearrange((hr.cpu().numpy()+1.0)*0.5, 'c h w -> h w c')
            hr_img = Image.fromarray(hr_img.astype(np.uint8))
            hr_img = hr_img.resize((image.shape[1], image.shape[0]))
            hr_img.save(save_path)
            index += 1    

def semseg(image, coco_categories):
    '''Semantic segmentation and generation of RGB masks

    :param image_path: _description_
    :param model: _description_
    :param colorizer: _description_
    :param color_map: _description_
    :param transform: _description_
    :param output_path: _description_, defaults to './mask_bank'
    '''

    with torch.no_grad():
        width = image.size[0]
        height = image.size[1]
        image = transform(image)
        image = np.asarray(image)
        images = torch.from_numpy(image.copy()).permute(2,0,1).cuda()

        batch_inputs = [{'image': images, 'height': height, 'width': width}]
        outputs = semseg_model.forward(batch_inputs,inference_task="sem_seg")
        matrix = outputs[-1]['sem_seg'].max(0)[1].cpu()

        gray_mask = Image.new("RGB", (matrix.shape[1], matrix.shape[0]))
        pixels = gray_mask.load()
        objects = []
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                color_index = matrix[i, j].item()
                pixels[j, i] = tuple([color_map[color_index]] * 3)
                current_obj = coco_categories[color_index]['category']
                if current_obj not in objects:
                    objects.append(current_obj)

        rgb_mask = gray_mask.convert('L')
        rgb_mask = np.array(rgb_mask)
        rgb_mask = np.transpose(colorizer(rgb_mask), (1,2,0))

        return objects, rgb_mask
            
def panoseg(image, categories):
    '''Panoptic segmentation to extract objects

    :param image_path: _description_
    :param model: _description_
    :param transform: _description_
    :param categories: _description_
    :return: _description_
    '''
    with torch.no_grad():
        image_ori = image.convert("RGB")
        width = image_ori.size[0]
        height = image_ori.size[1]
        image = transform(image_ori)
        image = np.asarray(image)
        image_ori = np.asarray(image_ori)
        images = torch.from_numpy(image.copy()).permute(2,0,1).cuda()

        batch_inputs = [{'image': images, 'height': height, 'width': width}]
        outputs = panoseg_model.forward(batch_inputs)

        seg_info = outputs[-1]['panoptic_seg'][1]

        items = []
        for item in seg_info:
            if item['isthing']:
                category_id = item['category_id']
                items.append(categories['thing'][category_id])
            else:
                category_id = item['category_id'] - len(categories['thing'])
                items.append(categories['stuff'][category_id])
        items.sort()
        return items

def get_objs_from_caption(caption, coco_categories, map_file, image=None, candi_objs=None, pano_categories=None, source='ic'):
    '''Extract objs and quantities from the given caption. 
    The parameter with the default value of None is only used when source is 'gt'

    :param caption: _description_
    :param coco_categories: _description_
    :param embed_model: _description_
    :param image_path: _description_, defaults to None
    :param candi_objs: _description_, defaults to None
    :param panoseg_model: _description_, defaults to None
    :param pano_categories: _description_, defaults to None
    :param transform: _description_, defaults to None
    :param source: _description_, defaults to 'ic'
    :raises ValueError: _description_
    :return: when source is 'gt': objs and num; when source is 'ic': objs, num and hasNum
    '''
    if source not in ['gt', 'ic']:
        raise ValueError(f"Invalid mode: {source}. Source must be 'gt' or 'ic'.")

    with torch.no_grad():
        target_objs = []
        ori_target_objs = []
        caption_objs = get_nouns(caption, pos_tagger)
        if source == 'gt':
            pano_objs = panoseg(image=image, categories=pano_categories)
            for obj in caption_objs:
                if obj['obj'].lower() not in black_list:
                    coco_category, color, map_file = category_to_coco(classifier=classifier, category=obj['obj'].lower(), coco_categories=coco_categories, map_file=map_file)
                    if coco_category in candi_objs:
                        data = {
                            'obj': coco_category,
                            'num': 0,
                            'color': color
                        }
                        if obj['hasNum']:
                            data['num'] = obj['num']
                        else:
                            data['num'] = pano_objs.count(coco_category)
                        ori_target_objs.append(obj['obj'])
                        target_objs.append(data)
            return target_objs, ori_target_objs, map_file
        else:
            ori_objs = copy.deepcopy(caption_objs)
            for obj in caption_objs:
                if obj['obj'].lower() not in black_list:
                    obj['obj'], _, map_file = category_to_coco(classifier=classifier, category=obj['obj'].lower(), coco_categories=coco_categories, map_file=map_file)
            return caption_objs, ori_objs, map_file

def inpaint(image, mask):
    return lama(image, mask)

def mask_dilate(ori_mask, bw_mask, target_color, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    exp_bw_mask = cv2.dilate(bw_mask, kernel, iterations=1)
    
    if target_color != None:
        for i in range(ori_mask.shape[0]):
            for j in range(ori_mask.shape[1]):
                if ori_mask[i][j].tolist() == target_color:
                    exp_bw_mask[i][j] = BLACKPIXEL
    return Image.fromarray(exp_bw_mask).convert('L')

def get_target_obj(image_path, mask, contains, kernel_size=70, output_path='./obj_mask/'):
    '''
    contains = [{'obj': 'bus', 'num': 3, 'color': [0, 128, 128]}]
    :param model: _description_
    :param image_path: _description_
    :param mask_path: _description_
    :param target_color: _description_, defaults to [255, 255, 255]
    :param output_path: _description_, defaults to './'
    '''

    save_path = os.path.join(output_path, os.path.splitext(os.path.basename(image_path))[0])
    mkdir(save_path)

    image = Image.open(image_path)
    objs = [item['obj'] for item in contains]
    colors = [item['color'] for item in contains]

    for target_obj, target_color in zip(objs, colors):
        inpaint_mask = np.zeros_like(mask)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i][j].tolist() != target_color and mask[i][j].tolist() in colors:
                    inpaint_mask[i][j] = WHITEPIXEL
                else:
                    inpaint_mask[i][j] = BLACKPIXEL
        inpaint_mask = mask_dilate(ori_mask=mask, bw_mask=inpaint_mask, target_color=target_color, kernel_size=kernel_size)
        inpaint_img = inpaint(image=image, mask=inpaint_mask)
        save_name = f'{target_obj}.png'
        inpaint_img.save(os.path.join(save_path, save_name))
    
    inpaint_mask = np.zeros_like(mask)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j].tolist() in colors:
                inpaint_mask[i][j] = WHITEPIXEL
            else:
                inpaint_mask[i][j] = BLACKPIXEL
    inpaint_mask = mask_dilate(ori_mask=mask, bw_mask=inpaint_mask, target_color=None, kernel_size=kernel_size)
    inpaint_img = inpaint(image=image, mask=inpaint_mask)
    save_name = 'background.png'
    inpaint_img.save(os.path.join(save_path, save_name))
    return save_path

def obj2mask(image_dir, contains, coco_categories):
    image_list = os.listdir(image_dir)
    for image_name in image_list:
        image_path = os.path.join(image_dir, image_name)
        image = Image.open(image_path)
        objs, mask = semseg(image=image, coco_categories=coco_categories)
        target_name = os.path.splitext(image_name)[0]
        if target_name != 'background':
            target_mask = np.ones_like(mask) * 255
            for item in contains:
                if item['obj'] == target_name:
                    target_color = item['color']
                    for i in range(target_mask.shape[0]):
                        for j in range(target_mask.shape[1]):
                            if mask[i][j].tolist() == target_color:
                                target_mask[i][j] = np.array(target_color)
                    break
        else:
            target_mask = mask
        target_mask = Image.fromarray(target_mask.astype(np.uint8))
        target_mask.save(image_path)

if __name__ == "__main__":
    args = parse_args()
    coco_categories = []
    with open('./id-category-color.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            coco_categories.append(data)

    image_bank_path = args.image_path
    mask_bank_path = args.mask_path
    gen_image_path = args.output_path
    obj_mask_path = args.obj_mask_path

    mkdir(mask_bank_path)
    mkdir(gen_image_path)
    mkdir(obj_mask_path)

    caption_path = args.caption_path
    target_objs_path = args.target_obj_path
    map_file_path = './category2coco.json'
    with open(map_file_path, 'r') as f:
        map_file = json.load(f)
    with open(caption_path, 'r') as f_r:
        data_list = json.load(f_r)

    with open(target_objs_path, 'w') as f_w:
        start_time = time.time()
        i = 0
        for key in tqdm(data_list.keys()):
            i += 1
            if i % 6 == 0:
                image_name = data_list[key]['img_id']
                base_name = os.path.splitext(image_name)[0]
                mask_dir = os.path.join(mask_bank_path, base_name)
                mkdir(mask_dir)
                caption = data_list[key]['caption']

                image_path = os.path.join(image_bank_path, image_name)
                image = Image.open(image_path)
                candi_objs, mask = semseg(image=image, coco_categories=coco_categories)
                mask = mask.astype(np.uint8)
                mask_img = Image.fromarray(mask)
                mask_name = base_name + '-0.png'
                mask_path = os.path.join(mask_dir, mask_name)
                mask_img.save(mask_path)

                try:
                    print('Extracting target objs from caption...')
                    target_objs, caption_objs, map_file = get_objs_from_caption(caption=caption, coco_categories=coco_categories, map_file=map_file, image=image, candi_objs=candi_objs, pano_categories=pano_categories, source='gt')
                    print(f'target_objs: {target_objs}')
                    print(f'caption_objs: {caption_objs}')


                    if len(target_objs) != 0:
                        sample_data = {
                            'name': image_name,
                            'tar_objs': target_objs,
                            'gt_objs': caption_objs,
                            'gt': caption
                        }
                        f_w.write(json.dumps(sample_data) + '\n')

                        print('Extracting target objs...')
                        obj_image_path = get_target_obj(image_path=image_path, mask=mask, contains=target_objs, kernel_size=50, output_path=obj_mask_path)
                        obj2mask(image_dir=obj_image_path, contains=target_objs, coco_categories=coco_categories)

                        print('Adjusting layout...')
                        editor(image_dir=obj_image_path, output_path=mask_dir, step=10, gen_num=49)

                        mask_list = os.listdir(mask_dir)
                        print('Generating new images...')
                        for item in tqdm(mask_list):
                            mask_base_name = os.path.splitext(item)[0]
                            mask_path = os.path.join(mask_dir, item)
                            output_dir = os.path.join(os.path.join(gen_image_path, base_name))
                            re_draw(image_path=mask_path, output_path=output_dir, num_samples=10)
                except Exception:
                    print('--------error--------')
                    traceback.print_exc()
                    print('---------------------')
                end_time = time.time()
                print(f'Total: {end_time - start_time}s')
                start_time = end_time
    print('Finish...')