import torch.nn as nn
import torch

class MMC(nn.Module):

    def __init__(self, loss_cfg):
        # TODO: run this code in gpu 1 or 0 (find out)
        super(MMC, self).__init__()
        try:
            self.is_mahalanobis = (loss_cfg.NAME == 'MMC')
            self.n_classes = loss_cfg.CLASSES
            self.means = (self.centers(loss_cfg.C_MM, loss_cfg.CLASSES, loss_cfg.FEAT_SPACE)).cuda()
            print('CONDUCTING MMC LOSS...')
        except AttributeError as err:
            print('CONDUCTING SCE LOSS...')
            self.is_mahalanobis = False

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


    def mmc_logits(self, x):
        '''
        Computes the Max-Mahalanobis Center loss.
        :param x: input, it is the output of the layer before logits
        :param means: pre-computed centers for each class.
        :param n_classes: number of classes in the dataset.
        :return:
        '''

        x_expand = torch.unsqueeze(x, dim=1).repeat((1, self.n_classes, 1))
        mean_expand = torch.unsqueeze(self.means, dim=0)
        logits = torch.sum( (x_expand - mean_expand)**2, dim=-1)
        return logits

        # x_expand = tf.tile(tf.expand_dims(x, axis=1), [1, num_class, 1])  # batch_size X num_class X num_dense
        # mean_expand = tf.expand_dims(means, axis=0)  # 1 X num_class X num_dense
        # logits = -tf.reduce_sum(tf.square(x_expand - mean_expand), axis=-1)  # batch_size X num_class
        # if use_ball == True:
        #     return logits
        # else:
        #     logits = logits - tf.reduce_max(logits, axis=-1, keepdims=True)  # Avoid numerical rounding
        #     logits = logits - tf.log(tf.reduce_sum(tf.exp(logits), axis=-1, keepdims=True))  # Avoid numerical rounding
        #     return logits


if __name__ == "__main__":
    from adv_package.attack.mahalanobis.mmc import centers
    test = MMC()
    test.mmc_loss(torch.rand(64, 10), means=centers(1,10,256))
