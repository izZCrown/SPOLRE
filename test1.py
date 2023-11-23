import sys
import os
sys.path.append('./tools/PITI')
sys.path.append('./tools/OpenSeeD')

# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import os
import sys
import logging

from PIL import Image
import numpy as np
np.random.seed(1)

import torch
from torchvision import transforms

from tools.OpenSeeD.utils.arguments import load_opt_command

from detectron2.data import MetadataCatalog
from detectron2.utils.colormap import random_color
from tools.OpenSeeD.openseed.BaseModel import BaseModel
from tools.OpenSeeD.openseed import build_model
from tools.OpenSeeD.utils.visualizer import Visualizer


logger = logging.getLogger(__name__)


def main(args=None):
    '''
    Main execution point for PyLearn.
    '''
    opt, cmdline_args = load_opt_command(args)
    for k, v in vars(cmdline_args).items():
        print(f'{k}, {v}')
    if cmdline_args.user_dir:
        absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
        opt['user_dir'] = absolute_user_dir

    # META DATA
    print('---------------')
    print(opt['WEIGHT'])
    print('---------------')
    opt['WEIGHT'] = '/home/wgy/multimodal/tools/ckpt/model_state_dict_swint_51.2ap.pt'
    pretrained_pth = os.path.join(opt['WEIGHT'])
    output_root = './output'
    image_pth = '/home/wgy/multimodal/MuMo/output/000000090208.jpg'

    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()

    t = []
    t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
    print('---------------')
    print(opt['WEIGHT'])
    print(t)
    print('---------------')
    transform = transforms.Compose(t)

    stuff_classes = ['zebra','antelope','giraffe','ostrich','sky','water','grass','sand','tree']
    stuff_colors = [random_color(rgb=True, maximum=255).astype(np.int).tolist() for _ in range(len(stuff_classes))]
    stuff_dataset_id_to_contiguous_id = {x:x for x in range(len(stuff_classes))}

    MetadataCatalog.get("demo").set(
        stuff_colors=stuff_colors,
        stuff_classes=stuff_classes,
        stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
    )
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(stuff_classes, is_eval=True)
    metadata = MetadataCatalog.get('demo')
    model.model.metadata = metadata
    model.model.sem_seg_head.num_classes = len(stuff_classes)

    with torch.no_grad():
        image_ori = Image.open(image_pth).convert("RGB")
        width = image_ori.size[0]
        height = image_ori.size[1]
        image = transform(image_ori)
        image = np.asarray(image)
        image_ori = np.asarray(image_ori)
        images = torch.from_numpy(image.copy()).permute(2,0,1).cuda()

        batch_inputs = [{'image': images, 'height': height, 'width': width}]
        outputs = model.forward(batch_inputs,inference_task="sem_seg")
        print(outputs)
        # visual = Visualizer(image_ori, metadata=metadata)
        # sem_seg = outputs[-1]['sem_seg'].max(0)[1]
        # demo = visual.draw_sem_seg(sem_seg.cpu(), alpha=0.5) # rgb Image

        # if not os.path.exists(output_root):
        #     os.makedirs(output_root)
        # demo.save(os.path.join(output_root, 'sem.png'))


if __name__ == "__main__":
    main()
    sys.exit(0)