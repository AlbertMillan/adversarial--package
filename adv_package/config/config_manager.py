from abc import ABCMeta, abstractmethod
from torch.utils.data import DataLoader
import sys

from ..attack.dataset import TORCH_CIFAR10, TORCH_CIFAR100
from .step import RawAttackStep, RawDefenseStep, AdvAttackStep, AdvDefenseStep, MixedAttackStep, MixedDefenseStep


class ConfigManager(metaclass=ABCMeta):
    _datasetDict = {
        'CIFAR10': TORCH_CIFAR10,
        'CIFAR100': TORCH_CIFAR100
    }

    _stepDict = {
        'RAW_ATT': RawAttackStep,
        'RAW_DEF': RawDefenseStep,
        'ADV_ATT': AdvAttackStep,
        'ADV_DEF': AdvDefenseStep,
        'BOTH_ATT': MixedAttackStep,
        'BOTH_DEF': MixedDefenseStep,
    }

    @abstractmethod
    def setDataset(self, arg):
        ''' Sets the dataset to be used.'''
        raise NotImplementedError

    @abstractmethod
    def setStep(self, att_cfg, model_cfg):
        ''' Sets the step conducted on each training iteration. '''
        raise NotImplementedError


class AttackManager(ConfigManager):

    def __init__(self, config):
        # TODO: DATASET MANAGER?
        dataset = self.setDataset(config.DATASET)
        target_data = (dataset.train_data if config.DATASET.TRAIN else self.dataset.test_data)
        self.batch_loader = DataLoader(target_data, batch_size=config.HYPERPARAMETERS.BATCH_SIZE)
        self.stepManager = self.setStep(config.ATTACK, config.MODELS)
        self.paths = config.PATHS
        self.print_freq = config.HYPERPARAMETERS.PRINT_FREQ

    def setDataset(self, arg):
        try:
            return self._datasetDict[arg.NAME](arg.DIR_PATH, arg.NORMALIZE, arg.CROP)
        except AttributeError as err:
            print('Error: Undefined variable in constructor {0}'.format(self.__class__.__name__))
            print(err)
            sys.exit(1)

    def setStep(self, att_cfg, model_cfg):
        try:
            return self._stepDict[att_cfg.STEP](att_cfg, model_cfg)
        except AttributeError as err:
            print('Error: Undefined variable in setStep {0}'.format(self.__class__.__name__))
            print(err)
            sys.exit(1)

    def run_pipeline(self):

        for i, (x, y) in enumerate(self.batch_loader):

            x = x.cuda()
            y = y.cuda()

            # Performs a step in the training procedure
            self.stepManager.takeTime()
            self.stepManager.step(x, y)
            self.stepManager.takeTime()

            if i % self.print_freq == 10:
                print(self.stepManager)

        print(' * Prec@1 {top1.avg:.3f}'.format(top1=self.stepManager.top1))
        return self.stepManager.lossMeter.avg, self.stepManager.top1.avg


class DefenseManager(ConfigManager):

    def __init__(self, config):
        # TODO: DATASET MANAGER?
        dataset = self.setDataset(config.DATASET)
        self.train_loader = DataLoader(dataset.train_data, batch_size=config.HYPERPARAMETERS.BATCH_SIZE)
        self.test_loader = DataLoader(dataset.test_data, batch_size=config.HYPERPARAMETERS.BATCH_SIZE)
        self.stepManager = self.setStep(config.ATTACK, config.MODELS, config.HYPERPARAMETERS)
        self.paths = config.PATHS

        # Hyperparameters configuration
        self.iterations = config.HYPERPARAMETERS.EPOCHS
        self.print_freq = config.HYPERPARAMETERS.PRINT_FREQ

        self.train_loss_hist = []
        self.train_acc1_hist = []
        self.train_acc5_hist = []

    def setDataset(self, arg):
        try:
            return self._datasetDict[arg.NAME](arg.DIR_PATH, arg.NORMALIZE, arg.CROP)
        except AttributeError as err:
            print('Error: Undefined variable in constructor {0}'.format(self.__class__.__name__))
            print(err)
            sys.exit(1)

    def setStep(self, att_cfg, model_cfg, hyperparam_cfg):
        try:
            return self._stepDict[att_cfg.STEP](att_cfg, model_cfg, hyperparam_cfg)
        except AttributeError as err:
            print('Error: Undefined variable in setStep {0}'.format(self.__class__.__name__))
            print(err)
            sys.exit(1)

    def run_pipeline(self):
        # TODO: Storing the state of the model during training

        best_pred = 0.0

        for epoch in range(self.iterations):
            self.stepManager.restart()

            for i, (x, y) in enumerate(self.train_loader):

                x = x.cuda()
                y = y.cuda()

                # Performs a step in the training procedure
                self.stepManager.takeTime()
                self.stepManager.step(x, y)
                self.stepManager.takeTime()

                if i % self.print_freq == 0:
                    print(self.stepManager.log(i, len(self.train_loader)))

            self.train_loss_hist.append(self.stepManager.lossMeter.avg)
            self.train_acc1_hist.append(self.stepManager.top1.avg)
            self.train_acc5_hist.append(self.stepManager.top5.avg)

        # print(' * Prec@1 {top1.avg:.3f}'.format(top1=self.stepManager.top1))
        return self.stepManager.lossMeter.avg, self.stepManager.top1.avg


