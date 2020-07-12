from .models import WideResNet, FullDenoiser

from abc import ABCMeta, abstractmethod
import torch
import torchvision.models as torchmodels
import shutil
import sys, os


class Model(metaclass=ABCMeta):

    @property
    @abstractmethod
    def parameters(self):
        ''' Returns relevant model parameters.'''
        raise NotImplementedError

    @staticmethod
    def _load_model(model, load_path=None, parallel=True):
        '''
        Returns model, in gpu if applicable, and with provided checkpoint if given.
        '''
        is_cuda = torch.cuda.is_available()
        n_gpus = torch.cuda.device_count()

        # Send to GPU if any
        if n_gpus > 1 and parallel:
            model = torch.nn.DataParallel(model).cuda()
            print(">>> SENDING MODEL TO PARALELL GPU...")
        elif is_cuda:
            model = model.cuda()
            print(">>> SENDING MODEL TO GPU...")

        # Load checkpoint
        if load_path:
            model = Model._load_checkpoint(model, load_path)
        #             print(">>> LOADING PRE-TRAINED MODEL:", load_path)

        return model

    @staticmethod
    def _load_checkpoint(model, filepath):
        """Load checkpoint from disk"""
        print("Loading Checkpoint:", filepath)
        if os.path.exists(filepath):
            state_dict = torch.load(filepath)
            model.load_state_dict(state_dict)
            print("Loaded checkpoint...")
            return model

        print("Failed to load model. Exiting...")
        sys.exit(1)

    @staticmethod
    def _save_checkpoint(state, save_dir, file_name):
        ''' Saves checkpoint to disk. '''
        directory = save_dir
        if not os.path.exists(directory):
            os.makedirs(directory)
        save_path = save_dir + file_name
        torch.save(state, file_name)

class StandardModel(Model, metaclass=ABCMeta):

    @abstractmethod
    def forward(self, x):
        '''
        Calls forward pass of the loaded model.
        '''
        raise NotImplementedError

    @abstractmethod
    def loss(self, x_batch, y_batch):
        '''
        Computes the loss function.
        '''
        raise NotImplementedError


class DenoiserModel(Model, metaclass=ABCMeta):
    ''' Template class for Denoiser models: different forward pass and loss during training and testing'''

    @abstractmethod
    def trainForward(self, x, x_adv):
        '''
        Calls forward pass of the loaded model during training.
        '''
        raise NotImplementedError

    @abstractmethod
    def testForward(self, x_input):
        '''
        Calls forward pass of the loaded model during training.
        '''
        raise NotImplementedError

    @abstractmethod
    def trainLoss(self, x_batch, y_batch):
        '''
        Computes the loss function.
        '''
        raise NotImplementedError

    def testLoss(self, x_batch, y_batch):
        '''
        Computes the loss function.
        '''
        raise NotImplementedError


    #     def _save_checkpoint(self, is_best, epoch, state, save_dir, base_name="chkpt"):
    # @staticmethod
    # def _save_checkpoint(is_best, state, save_dir, base_name="chkpt"):
    #     """Saves checkpoint to disk"""
    #     directory = save_dir
    #     filename = base_name + ".pth.tar"
    #     if not os.path.exists(directory):
    #         os.makedirs(directory)
    #     filename = directory + filename
    #     torch.save(state, filename)
    #     if is_best:
    #         shutil.copyfile(filename, directory + base_name + '__model_best.pth.tar')


class WrapperResNet(StandardModel):
    # TODO: Check on normalization conditions and if they match with the model pretrained version or not...
    # Maybe I can train the models, store them online, and download the model from some site...
    def __init__(self, depth, pretrained):
        resnet_model = {
            18: torchmodels.resnet18(pretrained=pretrained, progress=True),
            34: torchmodels.resnet34(pretrained=pretrained, progress=True),
            50: torchmodels.resnet50(pretrained=pretrained, progress=True),
            101: torchmodels.resnet101(pretrained=pretrained, progress=True),
            152: torchmodels.resnet152(pretrained=pretrained, progress=True),
        }

        assert depth in resnet_model, "ERROR: Depth {} does not exist for existing resnet models".format(depth)

        self.model = resnet_model[depth]

    @property
    def parameters(self):
        return self.model.parameters()

    def forward(self, x, x_adv=None):
        return self.model.forward(x)


class WrapperWideResNet(StandardModel):
    # Maybe I can train the models, store them online, and download the model from some site...
    def __init__(self, model_cfg):
        try:
            temp_model = WideResNet(model_cfg.DEPTH, 10, model_cfg.WIDEN_FACTOR, model_cfg.DROP_RATE)
            self.model = self._load_model(temp_model, model_cfg.CHKPT_PATH, model_cfg.PARALLEL)
            self.parallel = model_cfg.PARALLEL
            # self.save_dir = model_cfg.SAVE_CFG.SAVE_DIR
            # self.file_name = model_cfg.SAVE_CFG.SAVE_NAME
        except AttributeError as err:
            print('Error: Undefined variable in constructor {0}'.format(self.__class__.__name__))
            print(err)
            sys.exit(1)

    def _load_model(self, model, load_path, parallel):
        return super()._load_model(model, load_path, parallel)

    def save_model(self, save_dir, file_name):
        if self.parallel:
            super()._save_checkpoint(self.model.module.state_dict(), save_dir, file_name)
        else:
            super()._save_checkpoint(self.model.state_dict(), save_dir, file_name)

    def __call__(self, x):
        return self.model(x)

    @property
    def parameters(self):
        return self.model.parameters()

    def forward(self, x):
        return self.model(x)

    def loss(self, logits, y_batch):
        return self.model.module.loss(logits, y_batch) if self.parallel else self.model.loss(logits, y_batch)


class WrapperHGD(Model):

    def __init__(self, model_cfg):
        # Load target Model
        try:
            target_model_manager = ModelManager(model_cfg.TARGET)
            self.model = FullDenoiser(target_model_manager.getModel())
            self.model.denoiser = self._load_model(model_cfg.DENOISER_PATH, model_cfg.PARALLEL)
            self.parallel = model_cfg.PARALLEL
        except AttributeError as err:
            print('Error: Undefined variable in constructor {0}'.format(self.__class__.__name__))
            print(err)
            sys.exit(1)
        # self.save_dir = save_dir

    @property
    def parameters(self):
        return self.model.denoiser.parameters()

    def _load_model(self, load_path, parallel):
        return super()._load_model(self.model.denoiser, load_path, parallel)

    def save_model(self, save_dir, file_name):
        if self.parallel:
            super()._save_checkpoint(self.model.denoiser.module.state_dict(), save_dir, file_name)
        else:
            super()._save_checkpoint(self.model.denoiser.state_dict(), save_dir, file_name)

    # TODO: abstract with polymorphism
    def trainForward(self, x, x_adv):
        return self.model(x_adv, x)

    def testForward(self, x_input):
        return self.model(x_input)

    def trainLoss(self, logits_org, logits_smooth):
        # TODO: can I create generic method model.(module).train_loss
        return (self.model.module.train_loss(logits_org, logits_smooth) if self.parallel else \
                    self.model.train_loss(logits_org, logits_smooth))

    def testLoss(self, logits, y_batch):
        return (self.model.module.loss(logits, y_batch) if self.parallel else \
                    self.model.loss(logits, y_batch))



class ModelManager:
    # TODO: Implement Model Loader Class
    _modelDict = {
        'resnet': WrapperResNet,
        'wideresnet': WrapperWideResNet,
        'hgd': WrapperHGD
    }

    def __init__(self, model_config):
        self.cfg = model_config

    def getModel(self):
        # TODO: Create Iterative version for HGD...
        return self._modelDict[self.cfg.NAME.lower()](self.cfg)
