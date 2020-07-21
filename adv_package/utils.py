import torch
import time


def accuracy(logits, y, K=1):
    """Computes the precision@k for the specified values of k"""
    # Reshape to [N, 1]
    target = y.view(-1, 1)
    _, pred = torch.topk(logits, K, dim=1, largest=True, sorted=True)
    correct = (pred == target)

    return torch.sum(correct).float() / y.size(0)


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


class Tracker(object):

    def __init__(self):
        self.restart()
        self.time = time.time()

    def restart(self):
        self.lossMeter = AverageMeter()
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()
        self.batch_time = AverageMeter()

    def takeTime(self):
        self.batch_time.update(time.time() - self.time)
        self.time = time.time()

    def store(self, logits, loss, y_batch):
        prec1 = accuracy(logits.data, y_batch)
        prec5 = accuracy(logits.data, y_batch, K=5)
        self.top1.update(prec1.item(), y_batch.size(0))
        self.top5.update(prec5.item(), y_batch.size(0))
        self.lossMeter.update(loss.item(), y_batch.size(0))

    def log(self, i, n_batches, **kwargs):
        return ('Epoch: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, n_batches, batch_time=self.batch_time,
                    loss=self.lossMeter, top1=self.top1))

class EpochTracker(Tracker):

    def __init__(self, epoch):
        super().__init__()
        self.max_epoch = epoch

    def setEpoch(self, epoch):
        self.current = epoch

    def log(self, i, n_batches, is_training):
        if is_training:
            return self._log_training(i, n_batches)
        else:
            return self._log_testing(i, n_batches)

    def _log_training(self, i, n_batches):
        return ('Epoch: [{0}/{1}][{2}/{3}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    self.current, self.max_epoch, i, n_batches,
                    batch_time=self.batch_time, loss=self.lossMeter,
                    top1=self.top1))

    def _log_testing(self, i, n_batches):
        return super().log(i, n_batches)


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
