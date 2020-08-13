from abc import ABCMeta, abstractmethod
import time

import torch

from ..attack.attacker import IFGSM, MIM
from ..defense.optim import SGD, ADAM
from ..loader.model import ModelManager
from ..utils import EpochTracker


class Step(metaclass=ABCMeta):
    _attackDict = {
        'FGSM': IFGSM,
        'I-FGSM': IFGSM,
        'IFGSM': IFGSM,
        'PGD': IFGSM,
        'MIM': MIM
    }

    @abstractmethod
    def setAttack(self, model_cfg, att_cfg):
        ''' Creates Attack wrapper.'''
        raise NotImplementedError

    @abstractmethod
    def step(self, x_batch, y_batch):
        ''' Implements step.'''
        raise NotImplementedError


class StepManager(Step):
    # TODO: Implement Class to record train/test data
    _optimDict = {
        'SGD': SGD,
        'ADAM': ADAM
    }

    def __init__(self, att_cfg, models_cfg, max_iter):
        self.attack = self.setAttack(models_cfg.ADV_MODEL, att_cfg)
        self.threat_model = self.retrieve_model(models_cfg.THREAT_MODEL)
        self.optimManager = None

        self.tracker = EpochTracker(max_iter)

    def setAttack(self, model_cfg, att_cfg):
        return self._attackDict[att_cfg.NAME](model_cfg, att_cfg)

    def setOptimizer(self, optim_cfg):
        self.optimManager = self._optimDict[optim_cfg.NAME](optim_cfg, self.threat_model.model.parameters())

    @classmethod
    def retrieve_model(cls, model_cfg):
        modelManager = ModelManager(model_cfg)
        return modelManager.getModel()


    def step(self, x_batch, y_batch):
        if self.threat_model.model.training:
            self.trainStep(x_batch, y_batch)
        else:
            self.testStep(x_batch, y_batch)

    # Action used during testing. Ignored during training.
    def optimStep(self):
        self.optimManager.step()


class RawStep(StepManager):

    def trainStep(self, x_batch, y_batch):
        ''' Computes performance based raw data only.'''
        logits = self.threat_model.forward(x_batch)
        loss = self.threat_model.loss(logits, y_batch)

        loss.backward()

        self.tracker.store(logits, loss, y_batch)

        self.optimManager.step()

    def testStep(self, x_batch, y_batch):
        ''' Computes performance based raw data only.'''
        logits = self.threat_model.forward(x_batch)
        loss = self.threat_model.loss(logits, y_batch)

        self.tracker.store(logits, loss, y_batch)

class AdvStep(StepManager):

    def trainStep(self, x_batch, y_batch):
        ''' Computes performance based on adv data only.'''
        x_adv = self.attack.run(x_batch, y_batch)
        logits = self.threat_model.forward(x_adv)
        loss = self.threat_model.loss(logits, y_batch)

        loss.backward()

        self.tracker.store(logits, loss, y_batch)

        self.optimManager.step()

    def testStep(self, x_batch, y_batch):
        ''' Computes performance based on adv data only.'''
        x_adv = self.attack.run(x_batch, y_batch)
        logits = self.threat_model.forward(x_adv)
        loss = self.threat_model.loss(logits, y_batch)

        self.tracker.store(logits, loss, y_batch)

class MixedStep(StepManager):

    def trainStep(self, x_batch, y_batch):
        ''' Computes peformance based on raw and adv data.'''
        logits = self.threat_model.forward(x_batch)
        loss = self.threat_model.loss(logits, y_batch)
        loss.backward()

        self.tracker.store(logits, loss, y_batch)
        self.optimManager.step()

        x_adv = self.attack.run(x_batch, y_batch)
        logits = self.threat_model.forward(x_adv)
        loss = self.threat_model.loss(logits, y_batch)
        loss.backward()

        self.tracker.store(logits, loss, y_batch)
        self.optimManager.step()

    def testStep(self, x_batch, y_batch):
        ''' Computes peformance based on raw and adv data.'''
        logits = self.threat_model.forward(x_batch)
        loss = self.threat_model.loss(logits, y_batch)

        self.tracker.store(logits, loss, y_batch)

        x_adv = self.attack.run(x_batch, y_batch)
        logits = self.threat_model.forward(x_adv)
        loss = self.threat_model.loss(logits, y_batch)

        self.tracker.store(logits, loss, y_batch)

class AdvHGDStep(StepManager):
    ''' There is a difference on how the loss is computed during training and testing procedure.'''
    # TODO: Can I reduce the key difference here and abstract the method?
    def trainStep(self, x_batch, y_batch):
        # 1. Generate adversarial example.
        x_adv = self.attack.run(x_batch, y_batch)

        # KEY DIFFERENCE (logits) Assume training procedure for now...
        lg_smooth, lg_raw = self.threat_model.trainForward(x_batch, x_adv)
        loss = self.threat_model.trainLoss(lg_raw, lg_smooth)
        loss.backward()

        self.tracker.store(lg_smooth, loss, y_batch)
        self.optimManager.step()

    def testStep(self, x_batch, y_batch):
        x_adv = self.attack.run(x_batch, y_batch)

        logits = self.threat_model.testForward(x_adv)
        loss = self.threat_model.testLoss(logits, y_batch)

        self.tracker.store(logits, loss, y_batch)
        self.optimManager.zeroGrad()


class MixedHGDStep(StepManager):

    def trainStep(self, x_batch, y_batch):
        lg_smooth, lg_raw = self.threat_model.trainForward(x_batch, x_batch)
        loss = self.threat_model.trainLoss(lg_raw, lg_smooth)
        loss.backward()

        self.tracker.store(lg_smooth, loss, y_batch)
        self.optimManager.step()

        x_adv = self.attack.run(x_batch, y_batch)
        lg_smooth, lg_raw = self.threat_model.trainForward(x_batch, x_adv)
        loss = self.threat_model.trainLoss(lg_raw, lg_smooth)
        loss.backward()

        self.tracker.store(lg_smooth, loss, y_batch)
        self.optimManager.step()

    def testStep(self, x_batch, y_batch):
        logits = self.threat_model.testForward(x_batch)
        loss = self.threat_model.testLoss(logits, y_batch)

        self.tracker.store(logits, loss, y_batch)
        self.optimManager.zeroGrad()

        x_adv = self.attack.run(x_batch, y_batch)
        logits = self.threat_model.testForward(x_adv)
        loss = self.threat_model.testLoss(logits, y_batch)

        self.tracker.store(logits, loss, y_batch)
        self.optimManager.zeroGrad()
