import sys
sys.path.append('./tools/PITI')
sys.path.append('./tools/OpenSeeD')

import os
from argument import parser, print_args
from einops import rearrange
from PIL import Image
import cv2
import numpy as np
np.random.seed(1)
from util import mkdir, load_opt, Colorize, category_to_coco, pos_tag
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

from tools.PITI.pretrained_diffusion import dist_util, logger
from torchvision.utils import make_grid
from tools.PITI.pretrained_diffusion.script_util import create_model_and_diffusion
from tools.PITI.pretrained_diffusion.image_datasets_mask import get_tensor
from tools.PITI.pretrained_diffusion.train_util import TrainLoop
from tools.PITI.pretrained_diffusion.glide_util import sample
import time

args = parser()

def re_draw(image_path, output_path='./image_bank/', num_samples=1, sample_c=1.3, sample_step=100, device='cuda'):
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

def semseg(image_path, model, colorizer, color_map, transform, coco_categories, output_path='./mask_bank'):
    '''Semantic segmentation and generation of RGB masks

    :param image_path: _description_
    :param model: _description_
    :param colorizer: _description_
    :param color_map: _description_
    :param transform: _description_
    :param output_path: _description_, defaults to './mask_bank'
    '''
    mkdir(output_path)

    with torch.no_grad():
        image = Image.open(image_path).convert("RGB")
        width = image.size[0]
        height = image.size[1]
        image = transform(image)
        image = np.asarray(image)
        images = torch.from_numpy(image.copy()).permute(2,0,1).cuda()

        batch_inputs = [{'image': images, 'height': height, 'width': width}]
        outputs = model.forward(batch_inputs,inference_task="sem_seg")
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
        rgb_mask = Image.fromarray(rgb_mask.astype(np.uint8))
        ori_name = os.path.basename(image_path)
        save_name = os.path.splitext(ori_name)[0] + '-' + str(0) + '.png'
        save_path = os.path.join(output_path, save_name)
        rgb_mask.save(save_path)

        return objects
            
def panoseg(image_path, model, transform, categories):
    '''Panoptic segmentation to extract objects

    :param image_path: _description_
    :param model: _description_
    :param transform: _description_
    :param categories: _description_
    :return: _description_
    '''
    with torch.no_grad():
        image_ori = Image.open(image_path).convert("RGB")
        width = image_ori.size[0]
        height = image_ori.size[1]
        image = transform(image_ori)
        image = np.asarray(image)
        image_ori = np.asarray(image_ori)
        images = torch.from_numpy(image.copy()).permute(2,0,1).cuda()

        batch_inputs = [{'image': images, 'height': height, 'width': width}]
        outputs = model.forward(batch_inputs)

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

def load_draw_model(options, model_path, device='cuda'):
    '''_summary_

    :param options: _description_
    :param model_path: _description_
    :param device: _description_, defaults to 'cuda'
    :return: _description_
    '''
    model_ckpt = torch.load(model_path, map_location='cpu')
    model, _ = create_model_and_diffusion(**options)
    model.load_state_dict(model_ckpt, strict=True)
    return model.eval().to(device)

def load_seg_model(model_path, conf_file=None, categories_path='./id-category.jsonl', mode='sem', device='cuda'):
    '''load segmentation model

    :param model_path: _description_
    :param conf_file: _description_, defaults to None
    :param categories_path: _description_, defaults to './id-category.jsonl'
    :param mode: _description_, defaults to 'sem'
    :param categories: _description_, defaults to {}
    :param device: _description_, defaults to 'cuda'
    :raises ValueError: _description_
    :return: _description_
    '''
    if mode not in ['sem', 'pano']:
        raise ValueError(f"Invalid mode: {mode}. Mode must be 'sem' or 'pano'.")
    opt = load_opt(conf_file)
    opt['WEIGHT'] = model_path
    model = BaseModel(opt, build_model(opt)).from_pretrained(opt['WEIGHT']).eval().to(device)
    if mode == 'sem':
        stuff_classes = []
        stuff_colors = []
        stuff_dataset_id_to_contiguous_id = {}

        with open(categories_path, 'rb') as f:
            index = 0
            while True:
                try:
                    data = pickle.load(f)
                    stuff_classes.append(data['category'])
                    stuff_colors.append([data['id']] * 3)
                    stuff_dataset_id_to_contiguous_id[index] = data['id']
                    index += 1
                except:
                    break
        MetadataCatalog.get("demo").set(
            stuff_colors=stuff_colors,
            stuff_classes=stuff_classes,
            stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
        )
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(stuff_classes, is_eval=True)
        metadata = MetadataCatalog.get('demo')
        model.model.metadata = metadata
        model.model.sem_seg_head.num_classes = len(stuff_classes)
        return model, stuff_dataset_id_to_contiguous_id
    else:
        thing_classes = []
        stuff_classes = []

        with open(categories_path, 'rb') as f:
            while True:
                try:
                    data = pickle.load(f)
                    if data['id'] < 90:
                        thing_classes.append(data['category'])
                    else:
                        stuff_classes.append(data['category'])
        categories = [{'thing': thing_classes}, {'stuff': stuff_classes}]
        thing_colors = [random_color(rgb=True, maximum=255).astype(np.int).tolist() for _ in range(len(thing_classes))]
        stuff_colors = [random_color(rgb=True, maximum=255).astype(np.int).tolist() for _ in range(len(stuff_classes))]
        thing_dataset_id_to_contiguous_id = {x:x for x in range(len(thing_classes))}
        stuff_dataset_id_to_contiguous_id = {x+len(thing_classes):x for x in range(len(stuff_classes))}

        MetadataCatalog.get("demo").set(
            thing_colors=thing_colors,
            thing_classes=thing_classes,
            thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
            stuff_colors=stuff_colors,
            stuff_classes=stuff_classes,
            stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
        )
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(thing_classes + stuff_classes, is_eval=False)
        metadata = MetadataCatalog.get('demo')
        model.model.metadata = metadata
        model.model.sem_seg_head.num_classes = len(thing_classes + stuff_classes)
        return model, categories

def load_embed_model(model_name_or_path, device='cuda'):
    model = SentenceTransformer(model_name_or_path).to(device)
    return model

def get_objs_from_caption(image_path, caption, candi_objs, panoseg_model, transform, coco_categories, pano_categories, embed_model, source='gt'):
    if source not in ['gt', 'ic']:
        raise ValueError(f"Invalid mode: {source}. Mode must be 'gt' or 'ic'.")
    # TODO
    # 通过semseg得到img中包含哪几类obj（candi_objs）
    # 对caption进行分词，并将其映射到coco类别
    target_objs = []
    caption_objs = pos_tag(caption)
    pano_objs = panoseg(image_path=image_path, model=panoseg_model, transform=transform, categories=pano_categories)
    for obj in caption_objs:
        _, coco_category = category_to_coco(model=embed_model, category=obj['obj'], coco_categories=coco_categories)
        if coco_category in candi_objs:
            count = pano_objs.count(coco_category)
            data = {
                'obj': coco_category,
                'num': count
            }
            target_objs.append(data)
    return target_objs

    # 映射后如果属于target_objs，则保留，否则，舍弃。
    # 通过panoseg确定每一类obj的数量
    # 最后保留的objs将作为本图片要检测的target

    pass





if __name__ == "__main__":
    # TODO: 循环，首先调用semseg获取image的彩色mask并保存到seed_bank；
    # 不断从seed_bank中选择mask进行变换，变换后首先进行re_draw，并对生成的image进行panoseg以确定是否保持了原始caption；
    # 如果是，将新的mask存入seed_bank，并将生成的image存入image_bank以用于检测。
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========load model==========
    with open('./PITI_model_options.json', 'r') as f:
        options = json.load(f)
    print('loading base model...')
    # base_model_path = '/home/wgy/multimodal/tools/ckpt/base_mask.pt'
    # base_model = load_draw_model(options['base_model'], base_model_path, device=device)

    print('loading sample model...')
    # sample_model_path = '/home/wgy/multimodal/tools/ckpt/upsample_mask.pt'
    # sample_model = load_draw_model(options['sample_model'], sample_model_path, device=device)

    seg_model_path = '/home/wgy/multimodal/tools/ckpt/model_state_dict_swint_51.2ap.pt'
    t = []
    t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
    transform = transforms.Compose(t)

    categories_path = './id-category-embed.pkl'
    print('loading semseg model...')
    semseg_model, color_map = load_seg_model(model_path=seg_model_path, categories_path=categories_path, mode='sem', device=device)
    colorizer = Colorize(182)

    print('loading panoseg model...')
    panoseg_model, pano_categories = load_seg_model(model_path=seg_model_path, mode='pano', device=device)

    print('loading completed')
    # ==============================

    coco_categories = []
    with open('/home/wgy/multimodal/MuMo/id-category-embed.pkl', 'rb') as f:
        while True:
            try:
                data = pickle.load(f)
                coco_categories.append(data)
            except:
                break

    # image_path = '/home/wgy/multimodal/MuMo/output/000000090208.jpg'
    # items = panoseg(image_path=image_path, model=panoseg_model, transform=transform, categories=categories)
    image_path = '/home/wgy/multimodal/MuMo/output/000000090208.jpg'
    objects = semseg(image_path=image_path, model=semseg_model, colorizer=colorizer, color_map=color_map, transform=transform, coco_categories=coco_categories)
    print(objects)

    # image_path = '/home/wgy/multimodal/MuMo/mask_bank/test.png'
    # re_draw(image_path, num_samples=20, device=device)
    # embed_model_name = 'SpanBERT/spanbert-large-cased'
    # embed_model = load_embed_model(model_name_or_path=embed_model_name, device=device)
    # with open('/home/wgy/multimodal/MuMo/id-category.jsonl', 'r') as f, open('/home/wgy/multimodal/MuMo/id-category-embed.pkl', 'wb') as f1:
    #     for line in f:
    #         data = json.loads(line)
    #         index = data['id']
    #         category = data['category']
    #         embed = embed_model.encode([category])[0]
    #         sample_data = {
    #             'id': index,
    #             'category': category,
    #             'embed': embed
    #         }
    #         pickle.dump(sample_data, f1)
    
    # coco_category = category_to_coco(model=embed_model, category='parking timer', coco_categories=coco_categories)
    # print(coco_category)




