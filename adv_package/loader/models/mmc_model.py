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
            return out

        def forward(self, x):
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

# class MMC(nn.Module):
#
#     def __init__(self, loss_cfg):
#         super(MMC, self).__init__()
#         try:
#             self.is_mahalanobis = (loss_cfg.NAME == 'MMC')
#             self.n_classes = loss_cfg.CLASSES
#             self.means = (self.centers(loss_cfg.C_MM, loss_cfg.CLASSES, loss_cfg.FEAT_SPACE)).cuda()
#             print('CONDUCTING MMC LOSS...')
#         except AttributeError as err:
#             print('CONDUCTING SCE LOSS...')
#             self.is_mahalanobis = False
#
#     @staticmethod
#     def centers(c_mm, n_classes, d):
#         out = torch.zeros((n_classes, d))
#         out[0, 0] = 1
#
#         for i in range(1, n_classes):
#
#             for j in range(0, i):
#                 out[i, j] = -(1 / (n_classes - 1) + torch.dot(out[i], out[j])) / out[j, j]
#
#             out[i, i] = torch.sqrt((1 - torch.norm(out[i]) ** 2))
#
#         out *= c_mm
#         return out
#
#
#     def mmc_logits(self, x):
#         '''
#         Computes the Max-Mahalanobis Center loss.
#         :param x: input, it is the output of the layer before logits
#         :param means: pre-computed centers for each class.
#         :param n_classes: number of classes in the dataset.
#         :return:
#         '''
#
#         x_expand = torch.unsqueeze(x, dim=1).repeat((1, self.n_classes, 1))
#         mean_expand = torch.unsqueeze(self.means, dim=0)
#         logits = torch.sum( (x_expand - mean_expand)**2, dim=-1)
#         return logits
#
#         # x_expand = tf.tile(tf.expand_dims(x, axis=1), [1, num_class, 1])  # batch_size X num_class X num_dense
#         # mean_expand = tf.expand_dims(means, axis=0)  # 1 X num_class X num_dense
#         # logits = -tf.reduce_sum(tf.square(x_expand - mean_expand), axis=-1)  # batch_size X num_class
#         # if use_ball == True:
#         #     return logits
#         # else:
#         #     logits = logits - tf.reduce_max(logits, axis=-1, keepdims=True)  # Avoid numerical rounding
#         #     logits = logits - tf.log(tf.reduce_sum(tf.exp(logits), axis=-1, keepdims=True))  # Avoid numerical rounding
#         #     return logits
#
#
# if __name__ == "__main__":
#     from adv_package.attack.mahalanobis.mmc import centers
#     test = MMC()
#     test.mmc_loss(torch.rand(64, 10), means=centers(1,10,256))
