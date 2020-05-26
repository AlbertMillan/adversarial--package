# TODO: Create generic method to compute adversaries given appropriate variables.
import torch
import time
from ..utils import AverageMeter, accuracy


def _step(threat_model, x_batch, y_batch, losses, top1):
    # 2. Compute adversaries scores

    logits = threat_model.forward(x_batch)
    loss = threat_model.loss(logits, y_batch=y_batch)

    # 3. Compute adversarial accuracy
    prec1 = accuracy(logits.data, y_batch)
    top1.update(prec1.item(), x_batch.size(0))
    losses.update(loss.item(), x_batch.size(0))



def evaluate(threat_model, wrapper, batch_loader, include_raw=False, include_adv=True, print_freq=10):
    
    threat_model.model.eval()
    
    losses = AverageMeter()
    batch_time = AverageMeter()
    top1 = AverageMeter()

    end = time.time()

    for i, (x,y) in enumerate(batch_loader):

        x = x.cuda()
        y = y.cuda()

        if include_raw:
            _step(threat_model, x, y, losses, top1)

        if include_adv:
            x_adv = wrapper.run(x, y)
            _step(threat_model, x_adv, y, losses, top1)

        batch_time.update(time.time() - end)
        end = time.time()


        if i % print_freq == 0:
            print('Epoch: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(batch_loader), batch_time=batch_time,
                      loss=losses, top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return (losses.avg, top1.avg)  