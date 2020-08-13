import torch
import torch.nn as nn
import math
import sys
import time


class K_WTA(nn.Module):

    def __init__(self, gamma, in_features):
        super(K_WTA, self).__init__()
        self.k = int(gamma * in_features)

    def forward(self, x):
        start = time.time()
        tmpx = x.view(x.shape[0], -1)
        # Get topk smallest value: 1D vector smallest value per-sample
        topval = tmpx.topk(self.k, dim=-1)[0][:,-1]
        topval = topval.view(-1,1,1)
        mask = (topval>x)
        out = mask * x
        # return mask * x

        # # Flatten input completely.
        # base_shape = x.size()
        # # x_flat = torch.flatten(x)
        # x_flat = x.view(base_shape[0], -1)
        # idx = torch.topk(x_flat, k=self.k, dim=-1)[1]
        # mask = torch.zeros_like(x_flat)
        # sup = torch.arange(x.size(0))
        # mask[sup, idx[sup, :]] = 1
        #
        # out = x_flat * mask
        # out = out.view(base_shape)
        print('TIME: {}'.format(time.time()-start))
        return out


class ReLU(nn.Module):

    def __init__(self, *args):
        super(ReLU, self).__init__()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x)


class ActivationsManager(nn.Module):

    _activationDict = {
        'k-WTA': K_WTA,
        'ReLU': ReLU,
    }

    def __init__(self, activation_cfg):
        super(ActivationsManager, self).__init__()
        self.currentActivation = self._activationDict[activation_cfg.NAME](activation_cfg)

    def forward(self, *args, **kwargs):
        self.currentActivation(parameters_go_here)


if __name__ == '__main__':
    model = K_WTA(0.1, 100)
    model.cuda()
    model(torch.rand((10, 32, 32)))