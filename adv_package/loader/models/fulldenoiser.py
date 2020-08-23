import torch
import torch.nn as nn
import numpy as np
import sys

from .denoiser import Denoiser


# class DenoiseLoss(nn.Module):
#     def __init__(self, n, hard_mining=0, norm=False):
#         super(DenoiseLoss, self).__init__()

#         self.n = n
#         assert(hard_mining >= 0 and hard_mining <= 1)
#         self.hard_mining = hard_mining
#         self.norm = norm

#     def forward(self, x, y):
#         loss = torc


# class Loss(nn.Module):
#     def __init__(self, n, hard_mining=0, norm=False):
#         super(Loss, self).__init__()
# #         self.loss = DenoiseLoss(n, hard_mining, norm)
#         self.n = n

#     def forward(self, out_adv, out_org):
#         z = []
#         for i in range(len(out_adv)):
#             loss = torch.pow(torch.abs(out_adv[i] - out_org[i]), self.n) / self.n
#             loss = loss.mean()
#             z.append(loss)

# #         print("LOSS:", z)
# #         sys.exit()
#         return torch.stack(z).mean()


class FullDenoiser(nn.Module):

    def __init__(self, target_model, denoiser_cfg, n=1, hard_mining=0, loss_norm=False):
        super(FullDenoiser, self).__init__()

        # 1. Load Models
        self.target_model = target_model
        self.target_model.model.eval()
        self.denoiser = Denoiser(denoiser_cfg)

        self.crossEntropy = nn.CrossEntropyLoss()

    def forward(self, x_adv, x=None):
        # 1. Compute denoised image. Need to check this...
        noise = self.denoiser(x_adv)

        x_smooth = x_adv + noise

        out_adv = self.target_model(x_smooth)

        if self.training:
            assert x is not None, "ERROR: Original image not provided to HGD forward pass..."
            
            out_org = self.target_model(x)
#             sys.exit()
    
            return out_adv, out_org

        return out_adv

    def train_loss(self, logits_org, logits_smooth):
        return torch.sum(torch.abs(logits_smooth - logits_org), dim=1).mean()

    def loss(self, logits, y):
        return self.crossEntropy(logits, y)
