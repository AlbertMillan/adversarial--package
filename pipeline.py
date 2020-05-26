import torch
from torch.utils.data import DataLoader

from adv_package import attack, loader
from adv_package.utils import AverageMeter, accuracy
import attacks

import time, os, sys



class Pipeline:
    
    def __init__(self, ds_config, att_config, model_config, hp_config, options_config, path_config):
        
        adv_config = model_config.ADV_MODEL
        threat_config = model_config.THREAT_MODEL
        
        # Retrieve Dataset Objects
        data = attack.get_dataset(ds_config)
        target_data = (data.train_data if ds_config.TRAIN else data.test_data)
        
        # Create Data Loader for either train or test data
        self.batch_loader = DataLoader(target_data, batch_size=hp_config.BATCH_SIZE)
        
        # Set Options
        self.include_raw = options_config.RAW
        self.include_adv = options_config.ADV
        
        
        # Set hyperparameters
        self.print_freq = hp_config.PRINT_FREQ
        self.cuda = torch.cuda.is_available()
        
        # TODO: MODIFY FOR PRETRAINED MODELS
        
        adv_model = loader.retrieve_model(adv_config, ds_config.CLASSES)
        self.wrapper = attacks.retrieve(att_config, adv_model, ds_config.NORMALIZE)
        
        self.threat_model = loader.retrieve_model(threat_config, ds_config.CLASSES)
    
        print("Loaded MODELS...")

        
        
    def _step(self, x_batch, y_batch, losses, top1):
        # 2. Compute adversaries scores
        
        logits = self.threat_model.forward(x_batch)
        loss = self.threat_model.loss(logits, y_batch=y_batch)

        # 3. Compute adversarial accuracy
        prec1 = accuracy(logits.data, y_batch)
        top1.update(prec1.item(), x_batch.size(0))
        losses.update(loss.item(), x_batch.size(0))
        
    
    # TODO: move this step to the package and implement a simple function...
    def run_attack(self):
        self.threat_model.model.eval()
    
        losses = AverageMeter()
        batch_time = AverageMeter()
        top1 = AverageMeter()
        
        end = time.time()
        
        for i, (x,y) in enumerate(self.batch_loader):
            
            x = x.cuda()
            y = y.cuda()
            
            if self.include_raw:
                self._step(x, y, losses, top1)
            
            if self.include_adv:
                x_adv = self.wrapper.run(x, y)
                self._step(x_adv, y, losses, top1)

            batch_time.update(time.time() - end)
            end = time.time()
            
            
            if i % self.print_freq == 0:
                print('Epoch: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(self.batch_loader), batch_time=batch_time,
                          loss=losses, top1=top1))
                
        print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
        return (losses.avg, top1.avg)
    
    
    def run_attack_test(self):
        return attack.evaluate(self.threat_model, self.wrapper, self.batch_loader, \
                               self.include_raw, self.include_adv, self.print_freq)
        
        