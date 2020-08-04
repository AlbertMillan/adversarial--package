from abc import ABCMeta, abstractmethod
import torch



class EpsilonManager(object):

    @abstractmethod
    def getEps(self):
        ''' Returns epsilon values for each sample in x'''
        raise NotImplementedError


class AlphaManager(object):

    @abstractmethod
    def getAlpha(self):
        ''' Returns alpha value for each sample in x'''
        raise NotImplementedError


class ShiftManager(object):

    @abstractmethod
    def getShift(self):
        ''' Returns the initial displacement on the input (l-ball)'''
        raise NotImplementedError



class FixedEpsilon(EpsilonManager):

    def __init__(self, eps_config):
        self.eps = eps_config.VALUE

    def getEps(self, batch_size):
        if batch_size == None or batch_size == 0:
            raise ValueError('Invalid batch size given at \'{}\''.format(self.__class__.__name__))
        return torch.full((batch_size,1,1,1), (self.eps / 255.)).cuda()

    def __str__(self):
        return 'Epsilon: {0}/255 == {1}'.format(self.eps, self.eps/255.)


class RandomEpsilon(EpsilonManager):

    def __init__(self, eps_config):
        self.min_val = eps_config.MIN_VAL
        self.max_val = eps_config.MAX_VAL

    def setMinVal(self, val):
        self.min_val = val


    def setMaxVal(self, val):
        self.max_val = val

    def getEps(self, batch_size):
        '''
        Returns array of size 'batch_size' with random epsilon values within range [min_val, max_val]
        '''
        if batch_size == None or batch_size == 0:
            raise ValueError('Invalid batch size given at \'{}\''.format(self.__class__.__name__))
        if torch.cuda.is_available():
            return (torch.randint(low=self.min_val, high=self.max_val, size=(batch_size,1,1,1))).float().cuda() / 255.

        return (torch.randint(low=self.min_val, high=self.max_val, size=(batch_size,1,1,1))).float() / 255.

class FixedAlpha(AlphaManager):

    def __init__(self, alpha_cfg):
        self.alpha = alpha_cfg.VALUE

    def getAlpha(self, *args):
        return self.alpha / 255.

class DivisorAlpha(AlphaManager):

    def __init__(self, alpha_cfg):
        self.divisor = alpha_cfg.DIVISOR

    def getAlpha(self, eps):
        return eps / self.divisor


    # def setAlpha(self, eps):
    #     '''
    #     Returns alpha in tensor format
    #     '''
    #     alpha = eps / self.divisor
    #     if not torch.is_tensor(alpha):
    #         alpha = [alpha]
    #         if torch.cuda.is_available():
    #             alpha = torch.cuda.FloatTensor(alpha)
    #         else:
    #             alpha = torch.FloatTensor(alpha)
    #
    #     return alpha

class ZeroShift(ShiftManager):

    def getShift(self, *args):
       return 0.

class RandomShift(ShiftManager):

    def getShift(self, input_size, eps):
        random = None
        if torch.cuda.is_available():
            random = torch.rand(input_size).cuda()
        else:
            random = torch.rand(input_size)

        test = eps.view(random.size(0), 1, 1, 1)
        return random * 2. * eps.view(random.size(0), 1, 1, 1) - eps.view(random.size(0), 1, 1, 1)

