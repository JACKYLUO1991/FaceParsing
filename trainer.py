import os
import os.path as osp
import time
import torch
import datetime

import torch.nn as nn
from torchvision.utils import save_image
import numpy as np
import torch.nn.functional as F

from networks import get_model
from utils import *
from criterion import *
from lr_scheduler import WarmupPolyLR

from torch.utils.tensorboard import SummaryWriter


class Trainer(object):
    """Training pipline"""

    def __init__(self, data_loader, config):

        # Data loader
        self.data_loader = data_loader
        self.writer = SummaryWriter('runs/training')

        # Model hyper-parameters
        self.imsize = config.imsize
        self.parallel = config.parallel
        self.arch = config.arch

        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.total_iters = self.epochs * len(self.data_loader)

        self.classes = config.classes
        self.g_lr = config.g_lr
        self.momentum = config.momentum
        self.weight_decay = config.weight_decay
        self.pretrained_model = config.pretrained_model  # int type

        self.img_path = config.img_path
        self.label_path = config.label_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.sample_step = config.sample_step
        self.tb_step = config.tb_step

        # Path
        self.sample_path = osp.join(config.sample_path, self.arch)
        self.model_save_path = osp.join(
            config.model_save_path, self.arch)

        self.build_model()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

        self.lr_scheduler = WarmupPolyLR(
            self.g_optimizer, max_iters=self.total_iters, power=0.9, warmup_factor=1.0 / 3, warmup_iters=500, warmup_method='linear')

    def train(self):
        if self.pretrained_model:
            start = self.pretrained_model + 1
        else:
            start = 0

        criterion = CriterionAll()
        criterion.cuda()

        # Data iterator
        for epoch in range(start, self.epochs):
            self.G.train()

            for i_iter, batch in enumerate(self.data_loader):
                i_iter += len(self.data_loader) * epoch
                # lr = adjust_learning_rate(self.g_lr,
                #                           self.g_optimizer, i_iter, self.total_iters)
                self.lr_scheduler.step()

                imgs, labels, edges = batch
                size = labels.size()
                imgs = imgs.cuda()
                labels = labels.cuda()

                if self.arch == 'CE2P' or self.arch == 'FaceParseNet101':
                    edges = edges.cuda()
                    preds = self.G(imgs)
                    c_loss = criterion(preds, [labels, edges])

                    labels_predict = preds[0][-1]

                else:
                    labels = labels.view(size[0], 1, size[1], size[2])
                    oneHot_size = (size[0], self.classes, size[1], size[2])
                    labels_real = torch.cuda.FloatTensor(
                        torch.Size(oneHot_size)).zero_()
                    labels_real = labels_real.scatter_(
                        1, labels.data.long().cuda(), 1.0)

                    labels_predict = self.G(imgs)
                    c_loss = cross_entropy2d(
                        labels_predict, labels.long(), reduction='mean')

                self.reset_grad()
                c_loss.backward()
                self.g_optimizer.step()

                # scalr info on tensorboard
                if (i_iter + 1) % self.tb_step == 0:
                    self.writer.add_scalar(
                        'cross_entrophy_loss', c_loss.data, i_iter)
                    self.writer.add_scalar(
                        'learning_rate', self.g_optimizer.param_groups[0]['lr'], i_iter)

                # Sample images
                if (i_iter + 1) % self.sample_step == 0:
                    labels_sample = generate_label(
                        labels_predict, self.imsize)
                    save_image(denorm(labels_sample.data),
                               osp.join(self.sample_path, '{}_predict.png'.format(i_iter + 1)))

                print('iter={} of {} completed, loss={}'.format(
                    i_iter, self.total_iters, c_loss.data))

            torch.save(self.G.state_dict(), osp.join(
                self.model_save_path, '{}_G.pth'.format(str(epoch))))

    def build_model(self):
        self.G = get_model(
            self.arch, url='./networks/resnet101-imagenet.pth').cuda()

        if self.parallel:
            self.G = nn.DataParallel(self.G)
        # Loss and optimizer
        self.g_optimizer = torch.optim.SGD(filter(
            lambda p: p.requires_grad, self.G.parameters()), self.g_lr, self.momentum, self.weight_decay)

        """WarmUp learning strategy"""
        # TODO: omit

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(osp.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def reset_grad(self):
        self.g_optimizer.zero_grad()
