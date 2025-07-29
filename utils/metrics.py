import numpy as np
import torch
from pytorch_metric_learning import losses
from torch import nn
import torch.nn.functional as F


def _dice_loss(pred, target):
    smooth = 1e-5
    pred = F.softmax(pred, dim=1)
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3))
    dice = 1 - ((2 * inter + smooth) / (union + smooth))

    return dice.mean()


def evaluation(pred, gt):
    smooth = 1e-5
    intersection = np.sum(pred * gt)
    dice = (2 * intersection + smooth) / (np.sum(pred) + np.sum(gt) + smooth)
    iou = dice / (2 - dice)

    return iou, dice
