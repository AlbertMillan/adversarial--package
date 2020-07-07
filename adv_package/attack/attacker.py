from abc import ABCMeta, abstractmethod
import torch
import sys

from ..loader.model import ModelManager
from .gradientHelper import FixedEpsilon, RandomEpsilon, DivisorAlpha, ZeroShift, RandomShift

class Attack(metaclass=ABCMeta):

    # @abstractmethod
    # def compute_alpha(self, eps, max_iter):
    #     '''
    #     Returns adversarial constant 'alpha', indicating the extent to which an example
    #     changes in the direction of the gradient.
    #     '''
    #     raise NotImplementedError
    
    @abstractmethod
    def run(self, x_batch, y_batch, target=None):
        '''
        Returns an adversarial example for original input `x_batch` and true label
        `y_batch`. If `target` is not `None`, then the adversarial example should be
        targeted to be classified as `target`.
        '''
        raise NotImplementedError

    @classmethod
    def retrieve_model(cls, model_cfg):
        modelManager = ModelManager(model_cfg)
        return modelManager.getModel()

    @staticmethod
    def clampValidImage(x_batch):
        return x_batch.data.clamp_(min=0.0, max=1.0)

    @staticmethod
    def clampValidPerturbation(x_raw, x_adv, noise):
        return torch.min( torch.max(x_adv, (x_raw-noise)), (x_raw+noise))

    @staticmethod
    def checkValidAdversary(x_raw, x_adv, noise):
        low_bound = x_raw - noise
        high_bound = x_raw + noise
        try:
            if torch.all(x_adv.lt(low_bound)) or torch.all(x_adv.gt(high_bound)):
                raise Exception
        except Exception:
            print('ERROR: Adversarial noise is larger than epsilon limit value.')
            sys.exit(1)
        print('Valid adversarial example.')


class GradientAttack(Attack):

    _epsilonDict = {
        'RANDOM': RandomEpsilon,
        'FIXED': FixedEpsilon
    }

    _alphaDict = {
        'CONSTANT': DivisorAlpha
    }

    _shiftDict = {
        'RAW': ZeroShift,
        'SHIFT': RandomShift
    }

    def __init__(self, model_cfg, att_cfg):
        self.modelWrapper = self.retrieve_model(model_cfg)
        self.epsManager = self.setEpsilon(att_cfg.EPSILON)
        self.alphaManager = self.setAlpha(att_cfg.ALPHA)
        self.initManager = self.setShift(att_cfg.INIT)
        self.max_iter = att_cfg.MAX_ITER

    # TODO: Retrieve epsilon and alpha using polymorphism
    def setEpsilon(self, eps_cfg):
        try:
            return self._epsilonDict[eps_cfg.TYPE](eps_cfg)
        except Exception as err:
            print('ERROR: EPSILON.TYPE either not defined or unregistered.')
            print(err)
            raise

    def setAlpha(self, alpha_cfg):
        try:
            return self._alphaDict[alpha_cfg.TYPE](alpha_cfg)
        except Exception as err:
            print('ERROR: ALPHA.TYPE either not defined or unregistered.')
            print(err)
            raise

    def setShift(self, init_cfg):
        try:
            return self._shiftDict[init_cfg.TYPE]()
        except Exception as err:
            print('ERROR: INIT.TYPE either not defined or unregistered.')
            print(err)
            raise
            # sys.exit(1)


class IFGSM(GradientAttack):

    def __init__(self, model, att_cfg):
        super().__init__(model, att_cfg)

    def run(self, x_batch, y_batch):

        # Retrieve step size
        eps = self.epsManager.getEps(x_batch.size(0))
        alpha = self.alphaManager.getAlpha(eps)

        x = (self.initManager.getShift(x_batch.size(), eps) + x_batch).clone().detach().requires_grad_(True).cuda()
        # self.checkValidAdversary(x_batch, x, eps)
        x.data = self.clampValidImage(x)

        for _ in range(self.max_iter):

            logits = self.modelWrapper.forward(x)
            loss = self.modelWrapper.loss(logits, y_batch)

            loss.backward()

            # Get gradient
            noise = x.grad.data

            # Compute Adversary
            x.data = x.data + alpha * torch.sign(noise)

            # Clamp data between valid ranges
            x.data = self.clampValidImage(self.clampValidPerturbation(x_batch, x, eps) )

            x.grad.zero_()

        # torch.save(x, 'adv10_image_ts.pt')

        return x


class MIM(GradientAttack):

    def __init__(self, model, att_cfg):
        super().__init__(model, att_cfg)
        self.momentum = att_cfg.MOMENTUM

    def run(self, x_batch, y_batch):
        # Retrieve step size
        eps = self.epsManager.getEps(x_batch.size(0))
        alpha = self.alphaManager.getAlpha(self.epsManager.getEps())

        x = (self.initManager.getShift(x_batch.size(), eps) + x_batch).clone().detach().requires_grad_(True).cuda()
        # self.checkValidAdversary(x)
        x.data = self.clampValidImage(x)

        g = torch.zeros(x_batch.size(0), 1, 1, 1).cuda()

        for _ in range(self.max_iter):

            logits = self.modelWrapper.forward(x)
            loss = self.modelWrapper.loss(logits, y_batch)

            loss.backward()

            # Get gradient
            noise = x.grad.data

            g = self.momentum * g.data + noise / torch.mean(torch.abs(noise), dim=(1, 2, 3), keepdim=True)
            noise = g.clone().detach()

            # Compute Adversary
            x.data = x.data + alpha.view(alpha.size(0), 1, 1, 1) * torch.sign(noise)

            # Clamp data between valid ranges
            x.data = self.clampValidImage( self.clampValidPerturbation(x_batch, x, eps) )

            x.grad.zero_()

        return x
