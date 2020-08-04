from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn

# class Logits(metaclass=ABCMeta):
#
#     def forward(self, x):
#         ''' Computes logits given output of the penultimate layer (pre-logits). '''
#         raise NotImplementedError

class MahalanobisLogits(nn.Module):

    def __init__(self, loss_cfg, feat_dim):
        super(MahalanobisLogits, self).__init__()
        self.means = self.centers(loss_cfg.C_MM, loss_cfg.CLASSES, feat_dim)
        self.n_classes = loss_cfg.CLASSES

    @staticmethod
    def centers(c_mm, n_classes, d):
        out = torch.zeros((n_classes, d))
        out[0, 0] = 1

        for i in range(1, n_classes):

            for j in range(0, i):
                out[i, j] = -(1 / (n_classes - 1) + torch.dot(out[i], out[j])) / out[j, j]

            out[i, i] = torch.sqrt((1 - torch.norm(out[i]) ** 2))

        out *= c_mm
        return out

    def forward(self, x):
        x_expand = torch.unsqueeze(x, dim=1).repeat((1, self.n_classes, 1))
        mean_expand = torch.unsqueeze(self.means, dim=0)
        logits = torch.sum( (x_expand - mean_expand)**2, dim=-1)
        return logits


class StandardLogits(nn.Module):

    def __init__(self, loss_cfg, in_features):
        ''' Args contains input and output of the layer.'''
        self.fc = nn.Linear(in_features, loss_cfg.CLASSES)

    def forward(self, x):
        return self.fc(x)

