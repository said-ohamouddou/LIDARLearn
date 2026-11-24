"""
DELA — Generic helpers

Paper: CVPR 2023 (preprint arXiv:2306.10547)
Authors: Binjie Chen, Yunzhou Xia, Yu Zang, Cheng Wang, Jonathan Li
Source: https://github.com/Matrix-ASC/DeLA
License: MIT

Small utility helpers (parameter grouping, weight decay filters,
metric meters) shared by the DELA classification and segmentation
entry points.
"""
import torch

from utils.logger import print_log
from utils.checkpoint import load_checkpoint


def load_state(fn, **args):
    state = load_checkpoint(fn)
    for i in args.keys():
        if hasattr(args[i], "load_state_dict"):
            args[i].load_state_dict(state[i])
    return state


def save_state(fn, **args):
    state = {}
    for i in args.keys():
        item = args[i].state_dict() if hasattr(args[i], "state_dict") else args[i]
        state[i] = item
    torch.save(state, fn)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


class Metric():
    def __init__(self, num_classes=13, device=torch.device("cuda")):
        self.n = num_classes
        self.label = torch.arange(num_classes, dtype=torch.int64, device=device).unsqueeze(1)
        self.device = device
        self.reset()

    def reset(self):
        # pred == label == i for i in 0...num_classes
        self.intersection = torch.zeros((self.n,), dtype=torch.int64, device=self.device)
        # pred == i or label == i
        self.union = torch.zeros((self.n,), dtype=torch.int64, device=self.device)
        # label == i
        self.count = torch.zeros((self.n,), dtype=torch.int64, device=self.device)
        #
        self.acc = 0.
        self.macc = 0.
        self.iou = [0.] * self.n
        self.miou = 0.

    def update(self, pred, label):
        """
        pred:  NxC
        label: N
        """
        # CxN
        pred = pred.max(dim=1)[1].unsqueeze(0) == self.label
        label = label.unsqueeze(0) == self.label
        self.tmp_c = label.sum(dim=1)
        self.count += self.tmp_c  # label.sum(dim=1)
        self.intersection += (pred & label).sum(dim=1)
        self.union += (pred | label).sum(dim=1)

    def calc(self, digits=4):
        acc = self.intersection.sum() / self.count.sum()
        self.acc = round(acc.item(), digits)
        macc = self.intersection / self.count
        macc = macc.mean()
        self.macc = round(macc.item(), digits)
        iou = self.intersection / self.union
        self.iou = [round(i.item(), digits) for i in iou]
        miou = iou.mean()
        self.miou = round(miou.item(), digits)

    def print(self, str="", iou=True, digits=4):
        self.calc(digits)
        if iou:
            print_log(f"{str} acc: {self.acc} || macc: {self.macc} || miou: {self.miou} || iou: {self.iou}")
        else:
            print_log(f"{str} acc: {self.acc} || macc: {self.macc}")
