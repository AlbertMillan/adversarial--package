from abc import ABCMeta, abstractmethod
import time

from ..attack.attacker import IFGSM, MIM
from ..defense.optim import SGD, ADAM
from ..loader.model import ModelManager
from ..utils import AverageMeter, accuracy


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

    def __init__(self, att_cfg, models_cfg, **kwargs):
        self.attack = self.setAttack(models_cfg.ADV_MODEL, att_cfg)
        self.threat_model = self.retrieve_model(models_cfg.THREAT_MODEL)
        self.lossMeter = AverageMeter()
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()
        self.batch_time = AverageMeter()
        self.time = time.time()

    def setAttack(self, model_cfg, att_cfg):
        return self._attackDict[att_cfg.NAME](model_cfg, att_cfg)

    def restart(self):
        self.lossMeter = AverageMeter()
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()
        self.batch_time = AverageMeter()

    def takeTime(self):
        self.batch_time.update(time.time() - self.time)
        self.time = time.time()

    def store(self, logits, loss, y_batch):
        prec1 = accuracy(logits.data, y_batch)
        self.top1.update(prec1.item(), y_batch.size(0))
        self.lossMeter.update(loss.item(), y_batch.size(0))

    @classmethod
    def retrieve_model(cls, model_cfg):
        modelManager = ModelManager(model_cfg)
        return modelManager.getModel()

    def log(self, i, n_batches):
        return ('Epoch: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            i, n_batches, batch_time=self.batch_time,
            loss=self.lossMeter, top1=self.top1))


class DefenseStep(StepManager):
    _optimDict = {
        'SGD': SGD,
        'ADAM': ADAM
    }

    def __init__(self, att_cfg, models_cfg, optim_cfg):
        super().__init__(att_cfg, models_cfg)
        self.optimManager = self.setOptimizer(optim_cfg, self.threat_model.model.parameters())

    def setOptimizer(self, optim_cfg, model_param):
        return self._optimDict[optim_cfg.NAME](optim_cfg, model_param)

    def optimStep(self):
        self.optimManager.step()


class HGDStep(DefenseStep):
    ''' Class aiming to provide distinct functionality during training and testing a defense method.'''

    def __init__(self, att_cfg, models_cfg, optim_cfg):
        super().__init__(att_cfg, models_cfg, optim_cfg)

    def step(self, x_batch, y_batch):
        if self.threat_model.model.training:
            self.trainStep(x_batch, y_batch)
        else:
            self.testStep(x_batch, y_batch)


class RawAttackStep(StepManager):

    def step(self, x_batch, y_batch):
        ''' Computes performance based raw data only.'''
        logits = self.threat_model.forward(x_batch)
        loss = self.threat_model.loss(logits, y_batch)

        self.store(logits, loss, y_batch)


class AdvAttackStep(StepManager):

    def step(self, x_batch, y_batch):
        ''' Computes performance based on adv data only.'''
        x_adv = self.attack.run(x_batch, y_batch)
        logits = self.threat_model.forward(x_adv)
        loss = self.threat_model.loss(logits, y_batch)

        self.store(logits, loss, y_batch)


class MixedAttackStep(StepManager):

    def step(self, x_batch, y_batch):
        ''' Computes peformance based on raw and adv data.'''
        logits = self.threat_model.forward(x_batch)
        loss = self.threat_model.loss(logits, y_batch)

        self.store(logits, loss, y_batch)

        x_adv = self.attack.run(x_batch, y_batch)
        logits = self.threat_model.forward(x_adv)
        loss = self.threat_model.loss(logits, y_batch)

        self.store(logits, loss, y_batch)


class RawDefenseStep(DefenseStep):

    def step(self, x_batch, y_batch):
        ''' Computes performance based raw data only.'''
        logits = self.threat_model.forward(x_batch)
        loss = self.threat_model.loss(logits, y_batch)

        loss.backward()

        self.store(logits, loss, y_batch)

        self.optimManager.step()


class AdvDefenseStep(DefenseStep):

    def step(self, x_batch, y_batch):
        ''' Computes performance based on adv data only.'''
        x_adv = self.attack.run(x_batch, y_batch)
        logits = self.threat_model.forward(x_adv)
        loss = self.threat_model.loss(logits, y_batch)

        loss.backward()

        self.store(logits, loss, y_batch)

        self.optimManager.step()


class MixedDefenseStep(DefenseStep):

    def step(self, x_batch, y_batch):
        ''' Computes peformance based on raw and adv data.'''
        logits = self.threat_model.forward(x_batch)
        loss = self.threat_model.loss(logits, y_batch)
        loss.backward()

        self.store(logits, loss, y_batch)
        self.optimManager.step()

        x_adv = self.attack.run(x_batch, y_batch)
        logits = self.threat_model.forward(x_adv)
        loss = self.threat_model.loss(logits, y_batch)
        loss.backward()

        self.store(logits, loss, y_batch)
        self.optimManager.step()


class AdvHGDStep(HGDStep):
    ''' There is a difference on how the loss is computed during training and testing procedure.'''
    # TODO: Can I reduce the key difference here and abstract the method?
    def trainStep(self, x_batch, y_batch):
        # 1. Generate adversarial example.
        x_adv = self.attack.run(x_batch, y_batch)

        # KEY DIFFERENCE (logits) Assume training procedure for now...
        lg_smooth, lg_raw = self.threat_model.trainForward(x_adv, x_batch)
        loss = self.threat_model.trainLoss(lg_raw, lg_smooth)
        loss.backward()

        self.store(lg_smooth, loss, y_batch)
        self.optimManager.step()

    def testStep(self, x_batch, y_batch):
        x_adv = self.attack.run(x_batch, y_batch)

        logits = self.threat_model.testForward(x_adv)
        loss = self.threat_model.testLoss(logits)

        self.store(logits, loss, y_batch)
        self.optimManager.zeroGrad()


class MixedHGDStep(HGDStep):

    def trainStep(self, x_batch, y_batch):
        lg_smooth, lg_raw = self.threat_model.forward(x_batch, x_batch)
        loss = self.threat_model.loss(lg_raw, lg_smooth)
        loss.backward()

        self.store(lg_smooth, loss, y_batch)
        self.optimManager.step()

        x_adv = self.attack.run(x_batch, y_batch)
        lg_smooth, lg_raw = self.threat_model.forward(x_adv, x_batch)
        loss = self.threat_model.loss(lg_raw, lg_smooth)
        loss.backward()

        self.store(lg_smooth, loss, y_batch)
        self.optimManager.step()

    def testStep(self, x_batch, y_batch):
        logits = self.threat_model.testForward(x_batch)
        loss = self.threat_model.testLoss(logits)

        self.store(logits, loss, y_batch)
        self.optimManager.zeroGrad()

        x_adv = self.attack.run(x_batch, y_batch)
        logits = self.threat_model.testForward(x_adv)
        loss = self.threat_model.testLoss(logits)

        self.store(logits, loss, y_batch)
        self.optimManager.zeroGrad()
