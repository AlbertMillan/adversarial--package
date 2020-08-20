from abc import ABCMeta, abstractmethod
from torch.utils.data import DataLoader
from torch import optim
from torchsummary import summary
import sys

from ..utils import MetricTracker
from ..attack.dataset import TORCH_CIFAR10, TORCH_CIFAR100
from .step_manager import RawStep, AdvStep, MixedStep, RatioStep, RandomStep, AdvHGDStep, MixedHGDStep


class ConfigManager(metaclass=ABCMeta):
    _datasetDict = {
        'CIFAR10': TORCH_CIFAR10,
        'CIFAR100': TORCH_CIFAR100
    }

    _stepDict = {
        'RAW': RawStep,
        'ADV': AdvStep,
        'BOTH': MixedStep,
        'RATIO': RatioStep,
        'RANDOM': RandomStep,
        'ADV_HGD': AdvHGDStep,
        'BOTH_HGD': MixedHGDStep
    }

    @abstractmethod
    def setDataset(self, arg):
        ''' Sets the dataset to be used.'''
        raise NotImplementedError

    @abstractmethod
    def setStep(self, att_cfg, model_cfg):
        ''' Sets the step conducted on each training iteration. '''
        raise NotImplementedError

    @staticmethod
    def computeEpoch(batch_loader, stepManager, print_freq, is_training):
        for i, (x, y) in enumerate(batch_loader):

            x = x.cuda()
            y = y.cuda()

            # Performs a step in the training procedure
            stepManager.tracker.takeTime()
            stepManager.step(x, y)
            stepManager.tracker.takeTime()

            if i % print_freq == 0:
                print(stepManager.tracker.log(i, len(batch_loader), is_training))

        print(' * Prec@1 {top1.avg:.3f}'.format(top1=stepManager.tracker.top1))
        return stepManager.tracker.lossMeter.avg, stepManager.tracker.top1.avg, stepManager.tracker.top5.avg


class AttackManager(ConfigManager):

    def __init__(self, config):
        dataset = self.setDataset(config.DATASET)
        target_data = (dataset.train_data if config.DATASET.TRAIN else dataset.test_data)
        self.batch_loader = DataLoader(target_data, batch_size=config.HYPERPARAMETERS.BATCH_SIZE)
        self.stepManager = self.setStep(config.ATTACK, config.MODELS, None)
        self.paths = config.PATHS
        self.print_freq = config.HYPERPARAMETERS.PRINT_FREQ

    def setDataset(self, arg):
        try:
            return self._datasetDict[arg.NAME](arg.DIR_PATH, arg.NORMALIZE, arg.CROP)
        except AttributeError as err:
            print('Error: Undefined variable in constructor {0}'.format(self.__class__.__name__))
            print(err)
            sys.exit(1)

    def setStep(self, att_cfg, model_cfg, max_iter):
        try:
            return self._stepDict[att_cfg.STEP](att_cfg, model_cfg, max_iter)
        except AttributeError as err:
            print('Error: Undefined variable in setStep {0}'.format(self.__class__.__name__))
            print(err)
            sys.exit(1)

    def run_pipeline(self):
        self.stepManager.threat_model.model.eval()
        return self.computeEpoch(self.batch_loader, self.stepManager, self.print_freq,
                                 self.stepManager.threat_model.model.training)


class DefenseManager(ConfigManager):

    def __init__(self, config):
        dataset = self.setDataset(config.DATASET)
        self.train_loader = DataLoader(dataset.train_data, batch_size=config.HYPERPARAMETERS.BATCH_SIZE, shuffle=config.DATASET.SHUFFLE)
        self.test_loader = DataLoader(dataset.test_data, batch_size=config.HYPERPARAMETERS.BATCH_SIZE)
        self.stepManager = self.setStep(config.ATTACK, config.MODELS, config.HYPERPARAMETERS.EPOCHS)
        self.paths = config.PATHS

        # Hyperparameters configuration
        self.stepManager.setOptimizer(config.HYPERPARAMETERS.OPTIM)
        self.iterations = config.HYPERPARAMETERS.EPOCHS
        self.print_freq = config.HYPERPARAMETERS.PRINT_FREQ

        self.train_metrics = MetricTracker()
        self.test_metrics = MetricTracker()

    def setDataset(self, arg):
        try:
            return self._datasetDict[arg.NAME](arg.DIR_PATH, arg.CROP, arg.NORMALIZE)
        except AttributeError as err:
            print('Error: Undefined variable in constructor {0}'.format(self.__class__.__name__))
            print(err)
            sys.exit(1)

    def setStep(self, att_cfg, model_cfg, max_iter):
        try:
            return self._stepDict[att_cfg.STEP](att_cfg, model_cfg, max_iter)
        except AttributeError as err:
            print('Error: Undefined variable in setStep {0}'.format(self.__class__.__name__))
            print(err)
            sys.exit(1)

    def run_pipeline(self):
        # TODO: Add testing on each iteration of the model.

        best_pred = 0.0
#         summary(self.stepManager.threat_model.model, (3,32,32))
        print('Trainable Parameters:', sum(p.numel() for p in self.stepManager.threat_model.model.parameters() if p.requires_grad))

        for epoch in range(self.iterations):
            self.stepManager.tracker.setEpoch(epoch)
            self.stepManager.tracker.restart()
            self.stepManager.threat_model.model.train()

            self.computeEpoch(self.train_loader, self.stepManager, self.print_freq,
                              self.stepManager.threat_model.model.training)

            self.train_metrics.update(self.stepManager.tracker.lossMeter.avg,
                                      self.stepManager.tracker.top1.avg,
                                      self.stepManager.tracker.top5.avg)

            # Account for step performed on scheduler...
            self.stepManager.optimManager.schedulerStep()

            # Evaluation on test data
            self.stepManager.tracker.restart()
            test_loss, test_acc1, test_acc5 = self.eval()
            self.test_metrics.update(test_loss, test_acc1, test_acc5)

            if test_acc1 > best_pred:
                best_pred = test_acc1
                self.stepManager.threat_model.save_model(self.paths.SAVE_DIR, self.paths.SAVE_NAME)

        # print(' * Prec@1 {top1.avg:.3f}'.format(top1=self.stepManager.top1))
        return self.stepManager.tracker.lossMeter.avg, self.stepManager.tracker.top1.avg, self.stepManager.tracker.top5.avg


    def eval(self):
        self.stepManager.threat_model.model.eval()
        return self.computeEpoch(self.test_loader, self.stepManager, self.print_freq,
                                 self.stepManager.threat_model.model.training)
