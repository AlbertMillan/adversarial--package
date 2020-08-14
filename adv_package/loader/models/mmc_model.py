import torch
import torch.nn.functional as F
import numpy as np


def create_mmc_object(class_object):

    class MMC(class_object):

        def __init__(self, model_cfg):
            super().__init__(model_cfg)
            means = self.centers(model_cfg.LOSS.C_MM, self.nClasses, self.nChannels)
            self.register_buffer('means', means)


        @staticmethod
        def centers(c_mm, n_classes, d):
            out = torch.zeros((n_classes, d))
            out[0, 0] = 1

            for i in range(1, n_classes):
                for j in range(0, i):
                    out[i, j] = -(1 / (n_classes - 1) + torch.dot(out[i], out[j])) / out[j, j]

                out[i, i] = torch.sqrt( torch.abs(1 - torch.norm(out[i])**2) )

            out *= c_mm
            return out * 10

        def forward(self, x):
            # TODO: Move this out to the model and call the parent function.
            out = self.conv1(x)
            out = self.block1(out)
            out = self.block2(out)
            out = self.block3(out)
            out = self.relu(self.bn1(out))
            out = F.avg_pool2d(out, 8)
            out = out.view(-1, self.nChannels)

            # TODO: Can I implement this as a torch function (https://pytorch.org/docs/stable/notes/extending.html)
            # Max-mahalanobis center logits
            x_expand = torch.unsqueeze(out, dim=1).repeat((1, self.nClasses, 1))
            # mean_expand = torch.unsqueeze(self.centers())
            mean_expand = torch.unsqueeze(self.means, dim=0)
            sup = (x_expand - mean_expand) ** 2
            logits = -torch.sum(sup, dim=-1)
            return logits

        def loss(self, logits, y):
            # TODO: LOSS function used?
            y_true = torch.zeros((y.size(0), self.nClasses)).cuda()
            y_true[np.arange(0, y.size(0)), y] = -1
            loss = torch.sum(logits * y_true, dim=1)
            final_loss = torch.mean(loss)
            return final_loss

    return MMC
