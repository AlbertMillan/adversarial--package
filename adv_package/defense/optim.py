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

class Scheduler(metaclass=ABCMeta):

    def step(self):
        ''' Performs a step for the given scheduler. '''
        raise NotImplementedError

class NoScheduler(Scheduler):

    def __init__(self, *args):
        pass

    def step(self):
        pass

class ScheduleAtEpoch(Scheduler):

    def __init__(self, cfg, optimizer):
        self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.EPOCH, gamma=cfg.LR_DECAY)

    def step(self):
        self.scheduler.step()

class OptimizerStep(object):

    _schedulerDict = {
        'NONE': NoScheduler,
        'STEP': ScheduleAtEpoch,
        # 'PLATEAU': SheduleOnPlateau
    }

    def setScheduler(self, scheduler_cfg, optim):
        return self._schedulerDict[scheduler_cfg.NAME](scheduler_cfg, optim)

    def zeroGrad(self):
        self.optim.zero_grad()

    def step(self):
        self.optim.step()
        self.optim.zero_grad()

    def schedulerStep(self):
        self.schedulerManager.step()


class SGD(OptimizerStep):

    def __init__(self, optim_cfg, model_parameters):
        self.optim = optim.SGD(model_parameters,
                               lr=optim_cfg.LEARNING_RATE,
                               weight_decay=optim_cfg.WEIGHT_DECAY,
                               momentum=optim_cfg.MOMENTUM,
                               nesterov=optim_cfg.NESTEROV)

        self.schedulerManager = self.setScheduler(optim_cfg.SCHEDULER, self.optim)

class ADAM(OptimizerStep):

    def __init__(self, optim_cfg, model_parameters):
        self.optim = optim.Adam(model_parameters,
                                lr=optim_cfg.LEARNING_RATE,
                                weight_decay=optim_cfg.WEIGHT_DECAY)

        self.schedulerManager = self.setScheduler(optim_cfg.SCHEDULER, self.optim)

