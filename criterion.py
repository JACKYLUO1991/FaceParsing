# -*- coding: utf-8 -*-
# @Author: luoling
# @Date:   2019-12-06 10:41:34
# @Last Modified by:   luoling
# @Last Modified time: 2019-12-18 17:52:49

import torch
import torch.nn.functional as F
import torch.nn as nn


def cross_entropy2d(input, target, weight=None, reduction='none'):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht or w != wt:
        input = F.interpolate(input, size=(
            ht, wt), mode="bilinear", align_corners=True)

    # https://zhuanlan.zhihu.com/p/76583143
    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    # https://www.cnblogs.com/marsggbo/p/10401215.html
    loss = F.cross_entropy(
        input, target, weight=weight, reduction=reduction, ignore_index=250
    )

    return loss


def bootstrapped_cross_entropy2d(input, target, K=100000, weight=None, size_average=True):
    """High-performance semantic segmentation using very deep fully convolutional networks"""
    batch_size = input.size()[0]

    def _bootstrap_xentropy_single(input, target, K, weight=None, size_average=True):
        n, c, h, w = input.size()
        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(
            input, target, weight=weight, reduce=False, size_average=False, ignore_index=250
        )

        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(
            input=torch.unsqueeze(input[i], 0),
            target=torch.unsqueeze(target[i], 0),
            K=K,
            weight=weight,
            size_average=size_average,
        )

    return loss / float(batch_size)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1., ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    @staticmethod
    def make_one_hot(labels, classes):
        one_hot = torch.cuda.FloatTensor(labels.size()[0], classes, labels.size()[
                                         2], labels.size()[3]).zero_()
        target = one_hot.scatter_(1, labels.data, 1)
        return target

    def forward(self, output, target):
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()
        target = self.make_one_hot(
            target.unsqueeze(dim=1), classes=output.size()[1])
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (output_flat.sum() + target_flat.sum() + self.smooth))
        return loss


class CriterionAll(nn.Module):
    """Segmentation aware and Edge aware loss."""

    def __init__(self, alpha=20, ignore_index=255):
        super(CriterionAll, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.weighted_criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction='none')
        self.alpha = alpha

    def parsing_loss(self, preds, target):
        h, w = target[0].size(1), target[0].size(2)

        pos_num = torch.sum(target[1] == 1, dtype=torch.float)
        neg_num = torch.sum(target[1] == 0, dtype=torch.float)

        weight_pos = neg_num / (pos_num + neg_num)
        weight_neg = pos_num / (pos_num + neg_num)
        weights = torch.tensor([weight_neg, weight_pos])
        loss = 0

        # Edge-aware branch
        preds_edge = preds[1][0]
        scale_pred = F.interpolate(input=preds_edge, size=(h, w),
                                   mode='bilinear', align_corners=True)
        loss += F.cross_entropy(scale_pred, target[1],
                                weights.cuda(), ignore_index=self.ignore_index)

        # Segmentation-aware branch
        preds_parsing = preds[0]
        if isinstance(preds_parsing, list):
            for idx, pred_parsing in enumerate(preds_parsing):
                scale_pred = F.interpolate(input=pred_parsing, size=(h, w),
                                           mode='bilinear', align_corners=True)

                # A High-Efficiency Framework for Constructing Large-Scale Face Parsing Benchmark
                if idx == len(preds_parsing) - 1:  # Is that the last term ?
                    loss += (torch.mul(self.weighted_criterion(scale_pred, target[0]), torch.where(
                        target[1] == 0, torch.Tensor([1]).cuda(), torch.Tensor([1 + self.alpha]).cuda()))).mean()
                else:
                    loss += self.criterion(scale_pred, target[0])
        else:
            scale_pred = F.interpolate(input=preds_parsing, size=(h, w),
                                       mode='bilinear', align_corners=True)
            loss += self.criterion(scale_pred, target[0])
        return loss

    def forward(self, preds, target):
        loss = self.parsing_loss(preds, target)
        return loss
