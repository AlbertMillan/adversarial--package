from abc import ABCMeta, abstractmethod
import torch
import torchvision.models as torchmodels
from .models import WideResNet, FullDenoiser
import shutil
import sys, os


# TODO: Maybe I can create a forward method to handle whether it is loaded with DataParallel or on single GPU
# TODO: Create saver and loader functions within the class to load pretrained models
class Model(metaclass=ABCMeta):
    
    @abstractmethod   
    def get_model(self):
        '''
        Returns model. might not be necessary...
        '''
        raise NotImplementedError
    
    @abstractmethod
    def forward(self, x, x_adv=None):
        '''
        Calls forward pass of the loaded model.
        '''
        raise NotImplementedError
        
    
    def _load_model(self, model, load_path=None, parallel=True):
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
            model = self._load_checkpoint(model, load_path)
#             print(">>> LOADING PRE-TRAINED MODEL:", load_path)
            
        return model
    
    
    def _load_checkpoint(self, model, filepath):
        """Load checkpoint from disk"""
        print("Loading Checkpoint:", filepath)
        if os.path.exists(filepath):
            state_dict = torch.load(filepath)
            model.load_state_dict(state_dict)
            print("Loaded checkpoint...")
            return model
        
        print("Failed to load model. Exiting...")
        sys.exit(1)
    
    
#     def _save_checkpoint(self, is_best, epoch, state, save_dir, base_name="chkpt"):
    def _save_checkpoint(self, is_best, state, save_dir, base_name="chkpt"):
        """Saves checkpoint to disk"""
        directory = save_dir
        filename = base_name + ".pth.tar"
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = directory + filename
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, directory + base_name + '__model_best.pth.tar')


class WrapperResNet(Model):
    # TODO: Check on normalization conditions and if they match with the model pretrained version or not...
    # Maybe I can train the models, store them online, and download the model from some site...
    def __init__(self, depth, pretrained):
        resnet_model = {
            18:  torchmodels.resnet18(pretrained=pretrained, progress=True),
            34:  torchmodels.resnet34(pretrained=pretrained, progress=True),
            50:  torchmodels.resnet50(pretrained=pretrained, progress=True),
            101: torchmodels.resnet101(pretrained=pretrained, progress=True),
            152: torchmodels.resnet152(pretrained=pretrained, progress=True), 
        }
        
        assert depth in resnet_model, "ERROR: Depth {} does not exist for existing resnet models".format(depth)
        
        self.model = resnet_model[depth]
    
    
    def get_model(self):
        return self.model
    
    def forward(self, x, x_adv=None):
        return self.model.forward(x)
    

    
class WrapperWideResNet(Model):
    # TODO: Check on normalization conditions and if they match with the model pretrained version or not...
    # Maybe I can train the models, store them online, and download the model from some site...
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, load_path=None, parallel=None):
        self.model = WideResNet(depth, num_classes, widen_factor, dropRate)
        self.model = self._load_model(load_path, parallel)
        self.parallel = parallel
    
    def _load_model(self, load_path, parallel):
        return super()._load_model(self.model, load_path, parallel)
    
    def get_model(self):
        return self.model
    
    def forward(self, x, x_adv=None):
        return self.model(x)
        
    def loss(self, logits, y_batch):
        return (self.model.module.loss(logits, y_batch) if self.parallel else self.model.loss(logits, y_batch))
    
    

class WrapperHGD(Model):
    
    def __init__(self, target_model, load_path, parallel, isTrain, save_dir=None):
        self.model = FullDenoiser(target_model)
        self.model.denoiser = self._load_model(load_path, parallel)
        self.model.toggle_mode(isTrain)
        self.parallel = parallel
        self.save_dir = save_dir
        
    
    def _load_model(self, load_path, parallel):
        return super()._load_model(self.model.denoiser, load_path, parallel)
        
    def save_model(self, is_best):
        if self.parallel:
            super()._save_checkpoint(is_best, self.model.denoiser.module.state_dict(), self.save_dir)
        else:
            super()._save_checkpoint(is_best, self.model.denoiser.state_dict(), self.save_dir)
            
    
    def get_model(self):
        return self.model
    
    def forward(self, x_adv, x=None):
        if self.model.training:
            out = self.model(x_adv, x)
        else:
            out = self.model(x_adv)
        return out
    
    def loss(self, logits_org, logits_smooth=None, y_batch=None):
        if self.model.training:
            return (self.model.module.train_loss(logits_org, logits_smooth) if self.parallel else \
                    self.model.train_loss(logits_org, logits_smooth))
        else:
            return (self.model.module.loss(logits_org, y_batch) if self.parallel else \
                    self.model.loss(logits_org, y_batch))

        
    

def retrieve_model(config, n_classes):
    
    print(config)
    
    model_name = config.NAME.lower()
    model = None
    
    if model_name == 'resnet':
        model = WrapperResNet(config.DEPTH, config.PRETRAINED)
        
    if model_name == 'wideresnet':
        model = WrapperWideResNet(config.DEPTH, n_classes, config.WIDEN_FACTOR, config.DROP_RATE, config.CHKPT_PATH, config.PARALLEL)
    
    if model_name == 'hgd':
        # Get target model config
        target_model = retrieve_model(config.TARGET, n_classes)
        model = WrapperHGD(target_model.model, config.DENOISER_PATH, config.PARALLEL, config.TRAIN, config.SAVE_DIR)
        
    return model


if __name__ == '__main__':
    pass
    