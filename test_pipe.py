import torch
from torch.utils.data import DataLoader

from adv_package import attack, defense, loader
from adv_package.utils import AverageMeter, accuracy
import attacks

import time, os, sys



class Pipeline:
    
    def __init__(self, ds_config, att_config, model_config, hp_config, options_config, path_config):
        
        adv_config = model_config.ADV_MODEL
        train_config = model_config.TRAIN_MODEL
        
        # Retrieve Dataset Objects
        data = attack.get_dataset(ds_config)
        train_data = data.train_data
        test_data = data.test_data
        
        # Create Data Loader for either train or test data
        self.train_loader = DataLoader(train_data, batch_size=hp_config.BATCH_SIZE, shuffle=True)
        self.test_loader = DataLoader(test_data, batch_size=hp_config.BATCH_SIZE)
        
        
        # Set Options
        self.include_raw = options_config.RAW
        self.include_adv = options_config.ADV
        self.ttype = train_config.NAME
        
        # Set hyperparameters
        self.iterations = hp_config.EPOCHS
        self.print_freq = hp_config.PRINT_FREQ
        self.cuda = torch.cuda.is_available()
        
        # Create Adversarial Example Generator
        adv_model = loader.retrieve_model(adv_config, ds_config.CLASSES)
        self.wrapper = attacks.retrieve(att_config, adv_model, ds_config.NORMALIZE)
        
        # Train model
        self.train_model = loader.retrieve_model(train_config, ds_config.CLASSES)
    
        print("Loaded MODELS...")

        
        
    def _step(self, x_batch, y_batch, optimizer, losses, top1):
        # 2. Compute adversaries scores
        logits = self.train_model.forward(x_batch)
        loss = self.train_model.loss(logits, y_batch)

        # 3. Compute adversarial accuracy
        prec1 = accuracy(logits.data, y_batch)
        top1.update(prec1.item(), x_batch.size(0))
        losses.update(loss.item(), x_batch.size(0))
        
        # Compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        
        # Set grads to zero for new iter
        optimizer.zero_grad()
        
        
    def _hgd_step(self, x_batch, adv_batch, y_batch, optimizer, losses, top1):
        # 2. Compute adversaries scores
        logits_smooth, logits_org = self.train_model.forward(adv_batch, x_batch)
        loss = self.train_model.loss(logits_org, logits_smooth=logits_smooth)

        # 3. Compute adversarial accuracy
        prec1 = accuracy(logits_smooth.data, y_batch)
        top1.update(prec1.item(), x_batch.size(0))
        losses.update(loss.item(), x_batch.size(0))
        
        # Compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        
        # Set grads to zero for new iter
        optimizer.zero_grad()
        
        
    
    # TODO: move this step to the package and implement a simple function...
    def run_defense(self, optim_config):
    
        train_loss_hist = []
        train_acc_hist = []
        test_loss_hist = []
        test_acc_hist = []
        
        best_pred = 0.0
        
        end = time.time()
        
        for itr in range(self.iterations):
            
            optimizer = defense.retrieve_optimizer(optim_config, self.train_model.model.denoiser)
    
            losses = AverageMeter()
            batch_time = AverageMeter()
            top1 = AverageMeter()
            
        
            for i, (x,y) in enumerate(self.train_loader):

                x = x.cuda()
                y = y.cuda()

                if self.include_raw:
#                     self._step(x, y, optimizer, losses, top1)
                    self._hgd_step(x, x, y, optimizer, losses, top1)
                    

                if self.include_adv:
                    x_adv = self.wrapper.run(x, y)
#                     self._step(x_adv, y, optimizer, losses, top1)
                    self._hgd_step(x, x_adv, y, optimizer, losses, top1)

                batch_time.update(time.time() - end)
                end = time.time()


                if i % self.print_freq == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                              itr, i, len(self.train_loader), batch_time=batch_time,
                              loss=losses, top1=top1))
            
            
            attack.evaluate(self.train_model, self.wrapper, self.test_loader, False, True, 10)
            
            train_loss_hist.append(losses.avg)
            train_acc_hist.append(top1.avg)
            # test_loss_hist.append(test_loss)
            # test_acc_hist.append(test_prec1)
            
            is_best = (test_prec1 > best_pred)
            self.train_model.save_model(is_best)
            if is_best:
                best_pred = test_prec1
                
            
        print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
        return (losses.avg, top1.avg)
        
    
    # Same method as the one above, just abstracted from adv_package.
    def run_defense_test(self, optim_config):
        defense.adversarial_training(self.train_model, self.wrapper, self.train_loader, \
                                     self.test_loader, optim_config, self.iterations,   \
                                     self.ttype, self.include_raw, self.include_adv,    \
                                     self.print_freq)
        