import torch


class AverageMeter():
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricTracker(object):

    def __init__(self):
        self.loss_hist = []
        self.acc1_hist = []
        self.acc5_hist = []

    def update(self, loss, acc1, acc5):
        self.loss_hist.append(loss)
        self.acc1_hist.append(acc1)
        self.acc5_hist.append(acc5)


class Logger(object):

    def __init__(self, log_cfg, procedure, cfg_name):
        self.folder = log_cfg.DIR
        self.file_name = log_cfg.FILE
        self.path = self.folder + self.file_name
        self.procedure = procedure
        self.cfg_name = cfg_name

    def update(self, metrics):
        ''' Writes down results of experiment given final metrics object. '''
        # 1. Open file, if not created create a new one.
        with open(self.path, 'a+') as f:
            f.write(
                'TYPE: {0} \t CONFIG: {1} \t LOSS: {loss:.3f} \t TOP1: {acc1:.3f} \t TOP5: {acc5:.3f}\n'.format(
                    self.procedure, self.cfg_name, loss=metrics[0], acc1=metrics[1], acc5=metrics[2]
                )
            )


def accuracy(logits, y, K=1):
    """Computes the precision@k for the specified values of k"""
    # Reshape to [N, 1]
    target = y.view(-1, 1)

    _, pred = torch.topk(logits, K, dim=1, largest=True, sorted=True)
    correct = torch.eq(pred, target)

    return torch.sum(correct).float() / y.size(0)