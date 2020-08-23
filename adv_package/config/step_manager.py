from abc import ABCMeta, abstractmethod
import time, sys

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
        

class RatioStep(StepManager):
    
    def __init__(self, att_cfg, model_cfg, max_iter):
        super().__init__(att_cfg, model_cfg, max_iter)
        self.r = att_cfg.WEIGHT
    
    def trainStep(self, x_batch, y_batch):
        x_adv = self.attack.run(x_batch, y_batch)
        
        adv_logits = self.threat_model.forward(x_adv)
        adv_loss = self.threat_model.loss(adv_logits, y_batch)
        
        raw_logits = self.threat_model.forward(x_batch)
        raw_loss = self.threat_model.loss(raw_logits, y_batch)
        
        loss = self.r * raw_loss + (1. - self.r) * adv_loss
        loss.backward()
        
        # Append logits and predictions from one tensor with the other.
        total_logits = torch.cat((raw_logits, adv_logits), 0)
        total_y = torch.cat((y_batch, y_batch), 0)
        
        
        self.tracker.store(total_logits, loss, total_y)
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
        
        
class RandomStep(StepManager):
    
    def __init__(self, att_cfg, model_cfg, max_iter):
        super().__init__(att_cfg, model_cfg, max_iter)
        print(att_cfg)
        self.lamda = att_cfg.WEIGHT
        self.ratio = att_cfg.ADV_RATE
        
    def trainStep(self, x_batch, y_batch):
        
        # Compute indices of adversarial elements
        n_adv = int(x_batch.size(0) * self.ratio)
        idx = torch.randperm(x_batch.size(0))
        adv_idx = idx[:n_adv]
        raw_idx = idx[n_adv:]
        
        # Compute indices of ouput
        x_adv = x_batch[adv_idx]
        x_raw = x_batch[raw_idx]
        y_adv = y_batch[adv_idx]
        y_raw = y_batch[raw_idx]
        
        
        # Compute Adversarial Examples
        xn_adv = self.attack.run(x_adv, y_adv)
               
        # Concatenate adversarial vector and non-adversarial
        x_joint = torch.cat((xn_adv, x_raw), 0)
        y_joint = torch.cat((y_adv, y_raw), 0)
        
        logits = self.threat_model.forward(x_joint)
        persample_loss = self.threat_model.loss(logits, y_joint)
        adv_loss = torch.sum(persample_loss[:n_adv])
        clean_loss = torch.sum(persample_loss[n_adv:])
        
        loss = (clean_loss + self.lamda * adv_loss) / (x_batch.size(0) - n_adv + self.lamda * n_adv)
        loss.backward()
        
        self.tracker.store(logits, loss, y_joint)
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
        
        
class RawHGDStep(StepManager):
    
    def trainStep(self, x_batch, y_batch):
        lg_smooth, lg_raw = self.threat_model.trainForward(x_batch, x_adv)
        loss = self.threat_model.trainLoss(lg_raw, lg_smooth)
        loss.backward()

        self.tracker.store(lg_smooth, loss, y_batch)
        self.optimManager.step()
    
    def testStep(self, x_batch, y_batch):
        logits = self.threat_model.testForward(x_batch)
        loss = self.threat_model.testLoss(logits, y_batch)
        
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
