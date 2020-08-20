import torch
import torch.nn as nn

class LogitsMMC(nn.Module):
    
    def __init__(self, model_cfg, nChannels):
        super(LogitsMMC, self).__init__()
        self.nChannels = nChannels
        self.nClasses = model_cfg.LOSS.CLASSES
        means = self.centers(model_cfg.LOSS.C_MM, self.nClasses, self.nChannels)
        self.register_buffer('means', means)
        print('In MMC Logits...')
        
    @staticmethod
    def centers(c_mm, n_classes, d):
        out = torch.zeros((n_classes, d))
        out[0, 0] = 1

        for i in range(1, n_classes):
            for j in range(0, i):
                out[i, j] = -(1 / (n_classes - 1) + torch.dot(out[i], out[j])) / out[j, j]

            out[i, i] = torch.sqrt( torch.abs(1 - torch.norm(out[i])**2) )

        out *= c_mm
        return out * 10
    
    
    def forward(self, features):
        # TODO: Can I implement this as a torch function (https://pytorch.org/docs/stable/notes/extending.html)
        # Max-mahalanobis center logits
        x_expand = torch.unsqueeze(features, dim=1).repeat((1, self.nClasses, 1))
        # mean_expand = torch.unsqueeze(self.centers())
        mean_expand = torch.unsqueeze(self.means, dim=0)
        sup = (x_expand - mean_expand) ** 2
        logits = -torch.sum(sup, dim=-1)
        return logits
    
    def loss(self, logits, y):
        # TODO: LOSS function used?
        y_true = torch.zeros((y.size(0), self.nClasses)).cuda()
        y_true[torch.arange(0, y.size(0)), y] = -1
        loss = torch.sum(logits * y_true, dim=1)
        final_loss = torch.mean(loss)
        return final_loss
    
    

class LogitsStandard(nn.Module):
    
    def __init__(self, model_cfg, in_features):
        super(LogitsStandard, self).__init__()
        self.fc = nn.Linear(in_features, model_cfg.LOSS.CLASSES)
        self.crossEntropy = nn.CrossEntropyLoss(reduction=model_cfg.LOSS.REDUCTION)
        print('In Standard Logits...')
        
    def forward(self, x):
        return self.fc(x)
    
    def loss(self, logits, y_batch):
        return self.crossEntropy(logits, y_batch)



class LogitsManager(object):
    ''' Acts as an intermediate class used to select the correct logits Module. '''
    _logitsDict = {
        'SCE': LogitsStandard,
        'MMC': LogitsMMC,
    }
    
    def __init__(self, model_cfg, in_features):
        try:
            self.currentLogits = self._logitsDict[model_cfg.LOSS.NAME](model_cfg, in_features)
        except AttributeError as err:
            print('Logits Module could not be instantiated. Running Standard Logits...')
            self.currentLogits = LogitsStandard(model_cfg, in_features)
            
    @property
    def logits(self):
        return self.currentLogits
