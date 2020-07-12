import numpy as np
import torch


class AverageMeter():
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricTracker(object):

    def __init__(self):
        self.loss_hist = []
        self.acc1_hist = []
        self.acc5_hist = []

    def update(self, loss, acc1, acc5):
        self.loss_hist.append(loss)
        self.acc1_hist.append(acc1)
        self.acc5_hist.append(acc5)


def accuracy(logits, y, K=1):
    """Computes the precision@k for the specified values of k"""
    # Reshape to [N, 1]
    target = y.view(-1, 1)

    _, pred = torch.topk(logits, K, dim=1, largest=True, sorted=True)
    correct = torch.eq(pred, target)

    return torch.sum(correct).float() / y.size(0)