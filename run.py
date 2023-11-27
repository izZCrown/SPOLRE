import sys
sys.path.append('./tools/PITI')
sys.path.append('./tools/OpenSeeD')

import os
from argument import parser, print_args
from einops import rearrange
from PIL import Image, ImageChops
import cv2
import numpy as np
np.random.seed(1)
from util import mkdir, Colorize, category_to_coco, get_nouns, load_draw_model, load_seg_model, load_embed_model
import torch
from torchvision import transforms
from tools.OpenSeeD.utils.arguments import load_opt_command
from detectron2.data import MetadataCatalog
from detectron2.utils.colormap import random_color
from tools.OpenSeeD.openseed.BaseModel import BaseModel
from tools.OpenSeeD.openseed import build_model
from tools.OpenSeeD.utils.visualizer import Visualizer
import json
from sentence_transformers import SentenceTransformer
import pickle
from simple_lama_inpainting import SimpleLama
lama = SimpleLama()

from tools.PITI.pretrained_diffusion import dist_util, logger
from torchvision.utils import make_grid
from tools.PITI.pretrained_diffusion.script_util import create_model_and_diffusion
from tools.PITI.pretrained_diffusion.image_datasets_mask import get_tensor
from tools.PITI.pretrained_diffusion.train_util import TrainLoop
from tools.PITI.pretrained_diffusion.glide_util import sample
import time
from tqdm import tqdm
from transformers import pipeline
from layout_editor import editor

args = parser()

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

# print('loading embedding model...')
# embed_model_name = 'SpanBERT/spanbert-large-cased'
# embed_model = load_embed_model(model_name_or_path=embed_model_name, device=device)

print('loading classifier...')
if device == 'cuda':
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)
else:
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

print('loading models completed')
# ==============================

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
    # label = image.convert("RGB")
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
    # mkdir(output_path)

    with torch.no_grad():
        # image = Image.open(image_path).convert("RGB")
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
        # rgb_mask = Image.fromarray(rgb_mask.astype(np.uint8))
        # ori_name = os.path.basename(image_path)
        # save_name = os.path.splitext(ori_name)[0] + '-' + str(0) + '.png'
        # save_path = os.path.join(output_path, save_name)
        # rgb_mask.save(save_path)

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
        # image_ori = Image.open(image_path).convert("RGB")
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


def get_objs_from_caption(caption, coco_categories, map_file='./category2coco.json', image=None, candi_objs=None, pano_categories=None, source='ic'):
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

    target_objs = []
    caption_objs = get_nouns(caption)
    if source == 'gt':
        pano_objs = panoseg(image=image, categories=pano_categories)
        for obj in caption_objs:
            coco_category, color = category_to_coco(classifier=classifier, category=obj['obj'].lower(), coco_categories=coco_categories, map_file=map_file)
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
                target_objs.append(data)
        return target_objs
    else:
        for obj in caption_objs:
            obj['obj'], _ = category_to_coco(classifier=classifier, category=obj['obj'], coco_categories=coco_categories, map_file=map_file)
        return caption_objs

def inpaint(image, mask, kernel_size):
    exp_mask = np.zeros_like(mask)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    exp_mask = cv2.dilate(mask, kernel, iterations=1)
    exp_mask = Image.fromarray(exp_mask).convert('L')
    inpaint_img = lama(image, exp_mask)
    return inpaint_img

def get_target_obj(image_path, mask, contains, kernel_size=40, output_path='./obj_mask/'):
    '''从image中把mask中target_color的obj扣掉
    contains = [{'obj': 'bus', 'num': 3, 'color': [0, 128, 128]}]
    :param model: _description_
    :param image_path: _description_
    :param mask_path: _description_
    :param target_color: _description_, defaults to [255, 255, 255]
    :param output_path: _description_, defaults to './'
    '''
    # 先获得要扣去obj的黑白mask，白色部分为要扣去的部分
    black_pixel = np.array([0, 0, 0])
    white_pixel = np.array([255, 255, 255])
    save_path = os.path.join(output_path, os.path.splitext(os.path.basename(image_path))[0])
    mkdir(save_path)

    image = Image.open(image_path)
    # mask = np.array(Image.open(mask_path))
    objs = [item['obj'] for item in contains]
    colors = [item['color'] for item in contains]
    inpaint_mask = np.zeros_like(mask)

    for target_obj, target_color in zip(objs, colors):
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i][j].tolist() != target_color and mask[i][j].tolist() in colors:
                    inpaint_mask[i][j] = white_pixel
                else:
                    inpaint_mask[i][j] = black_pixel
        inpaint_img = inpaint(image=image, mask=inpaint_mask, kernel_size=kernel_size)
        save_name = f'{target_obj}.png'
        inpaint_img.save(os.path.join(save_path, save_name))
    
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j].tolist() in colors:
                inpaint_mask[i][j] = white_pixel
            else:
                inpaint_mask[i][j] = black_pixel
    inpaint_img = inpaint(image=image, mask=inpaint_mask, kernel_size=kernel_size)
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
    coco_categories = []
    # with open('/home/wgy/multimodal/MuMo/id-category-color-embed.pkl', 'rb') as f:
    #     while True:
    #         try:
    #             data = pickle.load(f)
    #             coco_categories.append(data)
    #         except:
    #             break
    with open('./id-category-color.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            coco_categories.append(data)

    image_bank_path = './image_bank'
    mask_bank_path = './mask_bank'
    gen_image_path = './gen_image'
    obj_mask_path = './obj_mask'

    mkdir(mask_bank_path)
    mkdir(gen_image_path)
    mkdir(obj_mask_path)

    caption_path = '/home/wgy/multimodal/MuMo/image_cation.json'
    target_objs_path = '/home/wgy/multimodal/MuMo/target_objs.json'
    with open(caption_path, 'r') as f_r:
        data_list = json.load(f_r)

    with open(target_objs_path, 'w') as f_w:
        for key in tqdm(data_list.keys()):
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

            print('Extracting target objs from caption...')
            target_objs = get_objs_from_caption(caption=caption, coco_categories=coco_categories, image=image, candi_objs=candi_objs, pano_categories=pano_categories, source='gt')

            if len(target_objs) != 0:
                sample_data = {
                    'name': image_name,
                    'target': target_objs,
                    'candi': candi_objs,
                    'caption': caption
                }
                json.dump(sample_data, f_w, indent=2)

                print('Extracting target objs...')
                obj_image_path = get_target_obj(image_path=image_path, mask=mask, contains=target_objs, output_path=obj_mask_path)
                obj2mask(image_dir=obj_image_path, contains=target_objs, coco_categories=coco_categories)

                print('Adjusting layout...')
                editor(image_dir=obj_image_path, output_path=mask_dir, step=5, gen_num=9)

                mask_list = os.listdir(mask_dir)
                print('Generating new images...')
                for item in tqdm(mask_list):
                    mask_base_name = os.path.splitext(item)[0]
                    mask_path = os.path.join(mask_dir, item)
                    output_dir = os.path.join(os.path.join(gen_image_path, base_name))
                    re_draw(image_path=mask_path, output_path=output_dir, num_samples=10)






        
        # for key in tqdm(data_list.keys()):
        #     image_name = data_list[key]['img_id']
        #     if image_name == '000000002592.jpg':
        #         caption = data_list[key]['caption']

        #         image_path = os.path.join(image_bank_path, image_name)
        #         image = Image.open(image_path)
        #         candi_objs, mask = semseg(image=image, model=semseg_model, colorizer=colorizer, color_map=color_map, transform=transform, coco_categories=coco_categories)
        #         # mask = Image.fromarray(mask.astype(np.uint8))
        #         # mask_path = os.path.join(mask_bank_path, image_name)
        #         # mask.save(mask_path)

        #         target_objs = get_objs_from_caption(caption=caption, coco_categories=coco_categories, embed_model=embed_model, image=image, candi_objs=candi_objs, panoseg_model=panoseg_model, pano_categories=pano_categories, transform=transform, source='gt')

        #         sample_data = {
        #             'name': image_name,
        #             'target': target_objs,
        #             'candi': candi_objs
        #         }
        #         json.dump(sample_data, f_w, indent=4)

        #         obj_image_path = get_target_obj(image_path=image_path, mask=mask.astype(np.uint8), contains=target_objs, output_path=output_path)
        #         obj2mask(image_dir=obj_image_path, contains=target_objs, model=semseg_model, colorizer=colorizer, color_map=color_map, transform=transform, coco_categories=coco_categories)
        #         break

        # image_dir = '/home/wgy/multimodal/MuMo/gen_mask/000000002592'
        # mask_list = os.listdir(image_dir)
        # output_dir = './gen_image'

        # for image_name in tqdm(mask_list):
        #     base_name = os.path.splitext(image_name)[0]
        #     image_path = os.path.join(image_dir, image_name)
        #     output_path = os.path.join(output_dir, base_name)
        #     re_draw(image_path=image_path, base_model=base_model, sample_model=sample_model, output_path=output_path, num_samples=10)







