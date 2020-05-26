from abc import ABCMeta, abstractmethod

class Attack(metaclass=ABCMeta):
    
    @abstractmethod   
    def compute_alpha(self, eps, max_iter):
        '''
        Returns adversarial constant 'alpha', indicating the extent to which an example
        changes in the direction of the gradient.
        '''
        raise NotImplementedError
    
    @abstractmethod
    def run(self, x_batch, y_batch, target=None):
        '''
        Returns an adversarial example for original input `x_batch` and true label
        `y_batch`. If `target` is not `None`, then the adversarial example should be
        targeted to be classified as `target`.
        '''
        raise NotImplementedError