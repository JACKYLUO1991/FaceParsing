# -*- coding: utf-8 -*-
# @Author: luoling
# @Date:   2019-12-02 11:19:20
# @Last Modified by:   luoling
# @Last Modified time: 2020-02-25 21:06:12

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms
import torch.nn.functional as F
from imutils.paths import list_images

import os
from PIL import Image
import numpy as np
import cv2 as cv
import argparse

from segmentation.parsenet18 import FaceParseNet34
from segmentation.parsenet50 import FaceParseNet50
from segmentation.dfanet import DFANet
from segmentation.ce2p import CE2P
from segmentation.dabnet import DABNet
from segmentation.unet import UNet
from segmentation.danet import DANet
from segmentation.parsenet import FaceParseNet101


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))

    def f(x): return x.split(prefix, 1)[-1] if x.startswith(prefix) else x

    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(
            pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)

    return model


def vis_parsing_maps(im, parsing_anno, color=[230, 50, 20]):
    # Colors for all 20 parts
    part_colors = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255],
                   [255, 204, 204], [102, 51, 0], [
                       255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153],
                   [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno_color = np.zeros(
        (parsing_anno.shape[0], parsing_anno.shape[1], 3))

    for pi in range(len(part_colors)):
        index = np.where(parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im = cv.addWeighted(cv.cvtColor(
        vis_im, cv.COLOR_RGB2BGR), 1.0, vis_parsing_anno_color, 0.5, 0)
    vis_im = Image.fromarray(cv.cvtColor(vis_im, cv.COLOR_BGR2RGB))

    return vis_im


def get_network(cfg, pretrained_path):
    if cfg.network == 'DANet':
        net = DANet(pretrained=False)
    elif cfg.network == 'UNet':
        net = UNet()
    elif cfg.network == 'FaceParseNet101':
        net = FaceParseNet101(pretrained=False)
    elif cfg.network == 'DABNet':
        net = DABNet()
    elif cfg.network == 'CE2P':
        net = CE2P(pretrained=False)
    elif cfg.network == 'DFANet':
        net = DFANet()
    else:
        raise NameError("Name error...")
    net.load_state_dict(pretrained_path)
    net.eval()

    return net


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='valitation with CelebAMask-HQ')
    parser.add_argument('--cpu', type=str2bool,
                        default='true', help='Use cpu inference')
    parser.add_argument('--size', type=int,
                        default=512, help='size of image')
    parser.add_argument('--pretrained_path', type=str,
                        default="./segmentation/DANet_47.pth", help="pretrained model's path")
    parser.add_argument('--network', type=str,
                        default="DANet", help="network's name")
    args = parser.parse_args()

    # Segmentation model loading
    pretrained_path = args.pretrained_path
    if args.cpu:
        device = 'cpu'
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage.cuda(device))

    net = get_network(args, pretrained_dict)
    net = net.to(device)

    if not os.path.exists("result/images"):
        os.makedirs("result/images")
    if not os.path.exists("result/renders"):
        os.makedirs("result/renders")

    for idx, img_path in enumerate(list_images("./result/images")):
        img = Image.open(img_path).convert("RGB")
        img = img.resize((args.size, args.size))  # 512

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img_ = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = net(img_)
            if 'FaceParseNet' in args.network or args.network == 'CE2P':
                output = output[0][-1]
            output = F.interpolate(
                output, (args.size, args.size), mode='bilinear', align_corners=True)  # [1, 19, 512, 512]
            parsing = np.squeeze(output.data.max(1)[1].cpu().numpy(), axis=0)
            fusing = vis_parsing_maps(img, parsing)
            fusing.save(os.path.join("./result/renders", f"{idx}.png"))

    # For GT Mask
    # for idx, (img_path, mask_path) in enumerate(zip(list_images("./result/images"), list_images("./result/masks"))):
    #     img = Image.open(img_path).convert("RGB").resize((args.size, args.size))
    #     mask = Image.open(mask_path).convert("L")
    #     fusing = vis_parsing_maps(img, np.array(mask))
    #     fusing.save(os.path.join("./result/renders", f"{idx}.png"))
