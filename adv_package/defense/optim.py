from abc import ABCMeta, abstractmethod
from torch import optim


# class Optimizer(Metaclass=ABCMeta):
#
#     @abstractmethod
#     def zeroGrad(self):
#         ''' Sets the stored gradients to zero.'''
#         raise NotImplementedError
#
#     @abstractmethod
#     def step(self):
#         ''' Performs a step for the given optimizer.'''
#         raise NotImplementedError

class OptimizerStep(object):

    def zeroGrad(self):
        self.optim.zero_grad()

    def step(self):
        self.optim.step()
        self.optim.zero_grad()

class SGD(OptimizerStep):

    def __init__(self, optim_cfg, model_parameters):
        self.optim = optim.SGD(model_parameters,
                               lr=optim_cfg.LEARNING_RATE,
                               weight_decay=optim_cfg.WEIGHT_DECAY,
                               momentum=optim_cfg.MOMENTUM,
                               nesterov=optim_cfg.NESTEROV)

class ADAM(OptimizerStep):

    def __init__(self, optim_cfg, model_parameters):
        self.optim = optim.Adam(model_parameters,
                                lr=optim_cfg.LEARNING_RATE,
                                weight_decay=optim_cfg.WEIGHT_DECAY)

