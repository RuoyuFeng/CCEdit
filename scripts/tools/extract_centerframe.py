'''
Usage:
python scripts/tools/extract_centerframe.py \
    --p_video assets/Samples/tshirtman.mp4 \
    --p_save outputs/centerframe/tshirtman.png \
    --orifps 18 \
    --targetfps 6 \
    --n_keyframes 17 \
    --length_long 512 \
    --length_short 512
'''

import argparse
import json
import os
import random

import einops
import torchvision
import cv2
import numpy as np
import torch
from pytorch_lightning import seed_everything
from torch import autocast

from scripts.sampling.util import (
    chunk,
    create_model,
    init_sampling,
    load_video,
    load_video_keyframes,
    model_load_ckpt,
    perform_save_locally_image,
    perform_save_locally_video,
)
from sgm.util import append_dims


def extract_centerframe(p_video, p_save, orifps, targetfps, n_keyframes, length_long, length_short):
    if n_keyframes % 2 == 0:
        print('WARNING: n_keyframes should be odd, but got {}'.format(n_keyframes))
    keyframes = load_video_keyframes(p_video, orifps, targetfps, n_keyframes)
    H, W = keyframes[0].shape[1:]
    if H >= W:
        h, w = length_long, length_short
    else:
        h, w = length_short, length_long
    # keyframes = load_video_keyframes(p_video, orifps, targetfps, n_keyframes, (h, w))

    centerframe = keyframes[n_keyframes // 2, :, :, :].unsqueeze(0)
    centerframe = torch.nn.functional.interpolate(centerframe, (h, w), mode='bilinear', align_corners=False)
    centerframe = (centerframe + 1) / 2.
    centerframe = torch.clamp(centerframe, 0, 1)

    # transfer to numpy and save
    centerframe = centerframe.squeeze(0).permute(1, 2, 0).cpu().numpy()[..., ::-1]
    # mkdir
    os.makedirs(os.path.dirname(p_save), exist_ok=True)
    cv2.imwrite(p_save, (centerframe * 255).astype(np.uint8))
    print('save to {}'.format(p_save))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--p_video', type=str, default='')
    parser.add_argument('--p_save', type=str, default='')
    parser.add_argument('--dir_video', type=str, default='')
    parser.add_argument('--dir_save', type=str, default='')
    parser.add_argument('--orifps', type=int, default=18)
    parser.add_argument('--targetfps', type=int, default=3)
    parser.add_argument('--n_keyframes', type=int, default=9)
    parser.add_argument('--length_short', type=int, default=384)
    parser.add_argument('--length_long', type=int, default=576)
    args = parser.parse_args()

    assert (args.p_video != '' and args.p_save != '' ) or \
        (args.dir_video != "" and args.dir_save != "args.dir_save"), \
        'source video must be specified'

    orifps = args.orifps
    targetfps = args.targetfps
    n_keyframes = args.n_keyframes

    if args.p_video != '':
        p_video = args.p_video
        p_save = args.p_save
        extract_centerframe(p_video, p_save, orifps, targetfps, n_keyframes, args.length_long, args.length_short)
    else:
        dir_video = args.dir_video
        dir_save = args.dir_save
        os.makedirs(dir_save, exist_ok=True)
        subdirs = os.listdir(dir_video)
        for subdir in subdirs:
            subdir_video = os.path.join(dir_video, subdir)
            if not os.path.isdir(subdir_video):
                continue
            subdir_save = os.path.join(dir_save, subdir)
            os.makedirs(subdir_save, exist_ok=True)
            files = os.listdir(subdir_video)
            for file in files:
                if not file.endswith('.mp4') or os.path.isdir(file):
                    continue
                p_video = os.path.join(subdir_video, file)
                p_save = os.path.join(subdir_save, file.replace('.mp4', '.png'))
                print('{} -> {}'.format(p_video, p_save))

                extract_centerframe(p_video, p_save, orifps, targetfps, n_keyframes, args.length_long, args.length_short)



    