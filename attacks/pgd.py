import numpy as np
import torch
import sys, os

from adv_package import attack


# class PGD(attack.Attack):
class PGD(attack.GradientAttack):
    
    
    def __init__(self, model, eps_config, alpha_config, max_iter, momentum=False, is_normalized=False):
        
        self.model = model
        self.eps_config = eps_config
        self.alpha_config = alpha_config
        self.max_iter = max_iter
        self.momentum = momentum
        self.is_normalized = is_normalized
        
    
    def gradient_step_size(self, eps_config, alpha_config, batch_size=None):
        if eps_config.TYPE == "RANDOM":
            eps = super().random_epsilon(eps_config.MIN, eps_config.MAX, batch_size) / 255.
        elif eps_config.TYPE == "FIXED":
            eps = eps_config.VALUE / 255.
        else:
            print("ERROR: Invalid EPSILON.TYPE parameter introduced.")
            sys.exit(1)
            
        alpha = super().compute_alpha(eps, alpha_config.DIVISOR)
        
        return alpha
            
    
    def clamp_tensor(self, x_batch, is_normalized):
        """ Clamps tensor x between valid ranges for the image (normalized or scaled range)"""
        if is_normalized:
            x_batch.data[:,0,:,:].clamp_(min=self.min_val[0], max=self.max_val[0])
            x_batch.data[:,1,:,:].clamp_(min=self.min_val[1], max=self.max_val[1])
            x_batch.data[:,2,:,:].clamp_(min=self.min_val[2], max=self.max_val[2])
        else:
            x_batch.data.clamp_(min=0.0, max=1.0)
            
        return x_batch.data
    
    
    def run(self, x_batch, y_batch, target=None):
        
        x = x_batch.clone().detach().requires_grad_(True).cuda()
        
        # Compute alpha. Alpha might vary depending on the type of normalization.
        alpha = self.gradient_step_size(self.eps_config, self.alpha_config, x_batch.size(0))
        
        # Set velocity for momentum
        if self.momentum:
            g = torch.zeros(x_batch.size(0), 1, 1, 1).cuda()
            
        
        for _ in range(self.max_iter):
            
            logits = self.model.forward(x)
            loss = self.model.loss(logits, y_batch)
            
            loss.backward()
            
            # Get gradient
            noise = x.grad.data
            
            # Momentum : You should not be using the mean here...
            if self.momentum:
                g = self.momentum * g.data + noise / torch.mean(torch.abs(noise), dim=(1,2,3), keepdim=True)
                noise = g.clone().detach()
            
            # Compute Adversary
            x.data = x.data + alpha.view(alpha.size(0),1,1,1) * torch.sign(noise)
            
            # Clamp data between valid ranges
            x.data = self.clamp_tensor(x, self.is_normalized)
            
            x.grad.zero_()
        
        return x
        