import torch
from .attack import Attack


class GradientAttack(Attack):
    # TODO: Normalization
    def random_epsilon(self, min_val, max_val, batch_size):
        '''
        Returns array of size 'batch_size' with random epsilon values within range [min_val, max_val]
        '''
        if torch.cuda.is_available():
            return (torch.randint(low=min_val, high=max_val, size=(batch_size,))).float().cuda()
        
        return (torch.randint(low=min_val, high=max_val, size=(batch_size,))).float()
    
    
    def compute_alpha(self, eps, divisor):
        '''
        Returns alpha
        '''
        alpha = eps / divisor
        if not torch.is_tensor(alpha):
            alpha = [alpha]
            if torch.cuda.is_available():
                alpha = torch.cuda.FloatTensor(alpha)
            else:
                alpha = torch.FloatTensor(alpha) 
        
        return alpha
        
    
    
    