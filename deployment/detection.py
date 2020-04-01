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

from PIL import Image
import numpy as np
import cv2 as cv
import argparse
from scipy.ndimage import gaussian_filter

from data import cfg_mnet, cfg_re50
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms
from layers.functions.prior_box import PriorBox

# Segmentation package
from segmentation.parsenet import FaceParseNet101

import time


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


def rot90(v):
    return np.array([-v[1], v[0]])


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

    # Only for hair segmentation
    # pi = 13
    # index = np.where(parsing_anno == pi)
    # vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im = cv.addWeighted(cv.cvtColor(
        vis_im, cv.COLOR_RGB2BGR), 1.0, vis_parsing_anno_color, 0.5, 0)
    vis_im = Image.fromarray(cv.cvtColor(vis_im, cv.COLOR_BGR2RGB))
    # vis_im_hsv = cv.cvtColor(vis_im, cv.COLOR_RGB2HSV)
    # tar_color = np.zeros_like(im)
    # tar_color[:, :, 0] = color[0]
    # tar_color[:, :, 1] = color[1]
    # tar_color[:, :, 2] = color[2]

    # tar_hsv = cv.cvtColor(tar_color, cv.COLOR_BGR2HSV)
    # vis_im_hsv[:, :, 0:1] = tar_hsv[:, :, 0:1]
    # changed = cv.cvtColor(vis_im_hsv, cv.COLOR_HSV2BGR)

    # def sharpen(img):
    #     from skimage.filters import gaussian

    #     img = img * 1.0
    #     gauss_out = gaussian(img, sigma=5, multichannel=True)

    #     alpha = 3
    #     img_out = (img - gauss_out) * alpha + img

    #     img_out = img_out / 255.0

    #     mask_1 = img_out < 0
    #     mask_2 = img_out > 1

    #     img_out = img_out * (1 - mask_1)
    #     img_out = img_out * (1 - mask_2) + mask_2
    #     img_out = np.clip(img_out, 0, 1)
    #     img_out = img_out * 255

    #     return np.array(img_out, dtype=np.uint8)

    # changed = sharpen(changed)
    # changed[parsing_anno != pi] = cv.cvtColor(
    #     im, cv.COLOR_RGB2BGR)[parsing_anno != pi]

    '''mask = np.zeros((512, 512, 1), dtype=np.uint8)
    face_parsing = (parsing_anno == 13)
    mask[:, :, 0] = np.where(face_parsing, 255, 0)

    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))
    mask = cv.erode(mask, kernel)
    _, contours, _ = cv.findContours(
        mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # cv.drawContours(im, contours, -1, (255, 255, 0), 5)
    # cv.imwrite("mask.png", im[:, :, ::-1])

    center_xy = (20, 20)
    dist = cv.pointPolygonTest(contours[0], center_xy, True)
    print(dist)'''

    # return changed
    return vis_im


celebahq = [0, 0, 0, 204, 0, 0, 76, 153, 0, 204, 204, 0, 51, 51, 255, 204, 0, 204, 0, 255, 255, 255, 204, 204, 102, 51,
            0,
            255, 0, 0, 102, 204, 0, 255, 255, 0, 0, 0, 153, 0, 0, 204, 255, 51, 153, 0, 204, 204, 0, 51, 0, 255, 153,
            51, 0, 204, 0]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Retinaface')
    parser.add_argument('-m', '--trained_model', default='./weights/Resnet50.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--network', default='resnet50',
                        help='Backbone network mobile0.25 or resnet50')
    parser.add_argument('--cpu', type=str2bool,
                        default='true', help='Use cpu inference')
    parser.add_argument('--confidence_threshold', default=0.02,
                        type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.4,
                        type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750,
                        type=int, help='keep_top_k')
    parser.add_argument('--vis_thres', default=0.9,
                        type=float, help='visualization_threshold')

    args = parser.parse_args()

    # torch.set_grad_enabled(False)
    cfg = None
    if args.network == 'mobile0.25':
        cfg = cfg_mnet
    elif args.network == 'resnet50':
        cfg = cfg_re50
    else:
        raise ValueError("Don't support this network!")
    net = RetinaFace(cfg=cfg, phase='test')
    # if args.cpu:
    #     pretrained_dict = torch.load(
    #         args.trained_model, map_location=lambda storage, loc: storage)
    # else:
    #     device = torch.cuda.current_device()
    #     pretrained_dict = torch.load(
    #         args.trained_model, map_location=lambda storage, loc: storage.cuda(device))
    # net.load_state_dict(pretrained_dict)
    # net.eval()
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')

    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    resize = 1  # Hyperparameter

    img_raw = cv.imread("demo.jpg", cv.IMREAD_COLOR)
    img = np.float32(img_raw)
    im_height, im_width, _ = img.shape

    scale = torch.Tensor(
        [img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    with torch.no_grad():
        loc, conf, landms = net(img)

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(
        0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(
        np.float32, copy=False)
    keep = py_cpu_nms(dets, args.nms_threshold)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:args.keep_top_k, :]
    ldmarks = landms[:args.keep_top_k, :].reshape(-1, 5, 2)

    img_rgb = cv.cvtColor(img_raw, cv.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb)

    # Segmentation model loading
    pretrained_path = './segmentation/ParseNet_120_G.pth'
    if args.cpu:
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage.cuda(device))

    s_net = FaceParseNet101(pretrained=False)
    s_net.load_state_dict(pretrained_dict)
    s_net.eval()
    s_net = s_net.to(device)

    for idx, (b, lm) in enumerate(zip(dets, ldmarks)):
        # Only support single human
        if idx == 1:
            break
        if b[4] < args.vis_thres:
            continue

        # "PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION"
        eye_avg = (lm[0] + lm[1]) * 0.5 + 0.5
        mouth_avg = (lm[3] + lm[4]) * 0.5 + 0.5
        eye_to_eye = lm[1] - lm[0]
        eye_to_mouth = mouth_avg - eye_avg
        x = eye_to_eye - rot90(eye_to_mouth)

        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = rot90(x)
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        zoom = 1024 / (np.hypot(*x) * 2)

        # Shrink.
        shrink = int(np.floor(0.5 / zoom))
        if shrink > 1:
            size = (int(np.round(
                float(img.size[0]) / shrink)), int(np.round(float(img.size[1]) / shrink)))
            img = img.resize(size, Image.ANTIALIAS)
            quad /= shrink
            zoom *= shrink

        # Crop.
        border = max(int(np.round(1024 * 0.1 / zoom)), 3)
        crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(
            np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0),
                min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Simulate super-resolution.
        superres = int(np.exp2(np.ceil(np.log2(zoom))))
        if superres > 1:
            img = img.resize(
                (img.size[0] * superres, img.size[1] * superres), Image.ANTIALIAS)
            quad *= superres
            zoom /= superres

        # Pad.
        pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(
            np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] -
                                                                       img.size[0] + border, 0),
               max(pad[3] - img.size[1] + border, 0))
        if max(pad) > border - 4:
            pad = np.maximum(pad, int(np.round(1024 * 0.3 / zoom)))
            img = np.pad(np.float32(
                img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.mgrid[:h, :w, :1]
            mask = 1.0 - np.minimum(np.minimum(np.float32(x) / pad[0], np.float32(
                y) / pad[1]), np.minimum(np.float32(w - 1 - x) / pad[2], np.float32(h - 1 - y) / pad[3]))
            blur = 1024 * 0.02 / zoom
            # TODO: 时间很长的bug!!!
            tic = time.time()
            img += (gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            print(time.time() - tic)
            img += (np.median(img, axis=(0, 1)) - img) * \
                   np.clip(mask, 0.0, 1.0)
            img = Image.fromarray(
                np.uint8(np.clip(np.round(img), 0, 255)), 'RGB')
            quad += pad[0:2]

        # Transform.
        img = img.transform((4096, 4096), Image.QUAD,
                            (quad + 0.5).flatten(), Image.BILINEAR)
        # TODO: 512 or 1024 ???
        img = img.resize((512, 512), Image.ANTIALIAS)
        img.save("img_raw.jpg")

        transform = transforms.Compose([
            # transforms.Resize(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img_ = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = s_net(img_)
            # ---- For edge-aware networks ---
            output = output[0][-1]
            # ---- For edge-aware networks ---
            output = F.interpolate(
                output, (512, 512), mode='bilinear', align_corners=True)
            parsing = np.squeeze(output.data.max(1)[1].cpu().numpy(), axis=0)

            out_img = Image.fromarray(parsing.astype(np.uint8))
            out_img.putpalette(celebahq)
            # # out_img.show()
            out_img.save("img_pred.png")
            # fusing = vis_parsing_maps(img, parsing)
            # fusing.save("img_pred.png")
            # cv.imwrite("img_pred.png", fusing)
