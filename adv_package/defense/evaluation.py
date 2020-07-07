import torch
import time
from ..utils import AverageMeter, accuracy
from ..attack.evaluate import evaluate


def _step(train_model, x_batch, y_batch, optimizer, losses, top1):
    
    # 2. Compute adversaries scores
    logits = train_model.forward(x_batch)
    loss = train_model.loss(logits, y_batch)

    # 3. Compute adversarial accuracy
    prec1 = accuracy(logits.data, y_batch)
    top1.update(prec1.item(), x_batch.size(0))
    losses.update(loss.item(), x_batch.size(0))

    # Compute gradient and do SGD step
    loss.backward()
    optimizer.step()

    # Set grads to zero for new iter
    optimizer.zero_grad()


def _hgd_step(train_model, x_batch, adv_batch, y_batch, optimizer, losses, top1):
        
    # 2. Compute adversaries scores
    logits_smooth, logits_org = train_model.forward(adv_batch, x_batch)
    loss = train_model.loss(logits_org, logits_smooth=logits_smooth)

    # 3. Compute adversarial accuracy
    prec1 = accuracy(logits_smooth.data, y_batch)
    top1.update(prec1.item(), x_batch.size(0))
    losses.update(loss.item(), x_batch.size(0))

    # Compute gradient and do SGD step
    loss.backward()
    optimizer.step()

    # Set grads to zero for new iter
    optimizer.zero_grad()



def adversarial_training(train_model, wrapper, train_loader, test_loader, optim_config, iterations, ttype, include_raw=False, include_adv=True, print_freq=10):

        train_loss_hist = []
        train_acc_hist = []
        test_loss_hist = []
        test_acc_hist = []
        
        best_pred = 0.0
        
        end = time.time()
        
        for itr in range(iterations):
            
            
            optimizer = retrieve_optimizer(optim_config, train_model.model.denoiser)
    
            losses = AverageMeter()
            batch_time = AverageMeter()
            top1 = AverageMeter()
            
            train_model.model.train()
        
            for i, (x,y) in enumerate(train_loader):

                x = x.cuda()
                y = y.cuda()

                if include_raw:
                    _hgd_step(train_model, x, x, y, optimizer, losses, top1)
                    

                if include_adv:
                    x_adv = wrapper.run(x, y)
                    _hgd_step(train_model, x, x_adv, y, optimizer, losses, top1)

                batch_time.update(time.time() - end)
                end = time.time()


                if i % print_freq == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                              itr, i, len(train_loader), batch_time=batch_time,
                              loss=losses, top1=top1))
            
            
            test_loss, test_prec = evaluate(train_model, wrapper, test_loader, include_raw, include_adv, print_freq)
            
            train_loss_hist.append(losses.avg)
            train_acc_hist.append(top1.avg)
            test_loss_hist.append(test_loss)
            test_acc_hist.append(test_prec)
            
            is_best = (test_prec > best_pred)
            train_model.save_model(is_best)
            if is_best:
                best_pred = test_prec
            
        return (train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist)
    
    
    
    

# def adversarial_training(train_model, wrapper, train_loader, test_loader, optim_config, iterations, ttype, include_raw=False, include_adv=True, print_freq=10):
    
#         train_loss_hist = []
#         train_acc_hist = []
#         test_loss_hist = []
#         test_acc_hist = []
        
#         best_pred = 0.0
        
#         end = time.time()
        
#         for itr in range(iterations):
            
#             if ttype == "HGD":
#                 optimizer = retrieve_optimizer(optim_config, train_model.model.denoiser)
#             else:
#                 optimizer = retrieve_optimizer(optim_config, train_model.model)
    
#             losses = AverageMeter()
#             batch_time = AverageMeter()
#             top1 = AverageMeter()
        
#             for i, (x,y) in enumerate(train_loader):

#                 x = x.cuda()
#                 y = y.cuda()

#                 if include_raw:
#                     if ttype == "HGD":
#                         _hgd_step(train_model, x, x_adv, y, optimizer, losses, top1)
#                     else:
#                         _step(train_model, x, y, optimizer, losses, top1)
                    

#                 if include_adv:
#                     x_adv = wrapper.run(x, y)
#                     if ttype == "HGD":
#                         _hgd_step(train_model, x, x, y, optimizer, losses, top1)
#                     else:
#                         _step(train_model, x, y, optimizer, losses, top1)

#                 batch_time.update(time.time() - end)
#                 end = time.time()


#                 if i % print_freq == 0:
#                     print('Epoch: [{0}][{1}/{2}]\t'
#                           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
#                               itr, i, len(train_loader), batch_time=batch_time,
#                               loss=losses, top1=top1))
            
            
#             test_loss, test_prec = evaluate(train_model, wrapper, test_loader, include_raw, include_adv, print_freq)
            
#             train_loss_hist.append(losses.avg)
#             train_acc_hist.append(top1.avg)
#             test_loss_hist.append(test_loss)
#             test_acc_hist.append(test_prec)
            
#             is_best = (test_prec > best_pred)
#             train_model.save_model(is_best)
#             if is_best:
#                 best_pred = test_prec
            
#         return (train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist)