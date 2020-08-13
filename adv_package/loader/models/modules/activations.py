import torch
import torch.nn as nn
import math
import sys
import time


class K_WTA1D(nn.Module):

    def __init__(self, gamma):
        super(K_WTA1D, self).__init__()
        self.gamma = gamma

    def forward(self, x):
#         start = time.time()
        layer_size = x.shape[1] * x.shape[2] * x.shape[3]
        tmpx = x.view(x.shape[0], -1)
        # Get topk smallest value: 1D vector smallest value per-sample
        topval = tmpx.topk(int(self.gamma * layer_size), dim=-1)[0][:,-1]
        topval = topval.view(-1,1,1,1)
        mask = (topval>x).to(x)
        out = mask * x

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
#         print('TIME: {}'.format(time.time()-start))
        return out



class K_WTA2D(nn.Module):

    def __init__(self, gamma):
        super(K_WTA2D, self).__init__()
        self.gamma = gamma

    def forward(self, x):
        layer_size = x.size(-1) * x.size(-2)
        tmpx = x.view(x.shape[0], x.shape[1], -1)

        # Get topk smallest value: 1D vector smallest value per-sample
        topval = tmpx.topk(int(self.gamma * layer_size), dim=-1)[0][:,:,-1]
        topval = topval.expand(x.shape[2], x.shape[3], x.shape[0], x.shape[1]).permute(2,3,0,1)
        mask = (topval>x).to(x)
        out = mask * x

        return out


class ReLU(nn.Module):

    def __init__(self, *args):
        super(ReLU, self).__init__()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x)


class ActivationsManager(nn.Module):

    _activationDict = {
        'k-WTA1D': K_WTA1D,
        'k-WTA2D': K_WTA2D,
        'ReLU': ReLU,
    }

    def __init__(self, activation_cfg):
        super(ActivationsManager, self).__init__()
        try:
            self.currentActivation = self._activationDict[activation_cfg.NAME](activation_cfg.GAMMA)
        except AttributeError as err:
            print('K-WTA could not be executed (missing parameter in config file?). Running ReLU...')
            self.currentActivation = ReLU()

    def forward(self, x):
        return self.currentActivation(x)


if __name__ == '__main__':
    model = K_WTA(0.1, 100)
    model.cuda()
    model(torch.rand((10, 32, 32)))