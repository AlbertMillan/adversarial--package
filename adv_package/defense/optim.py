from torch import optim


def adam(model, lr, weight_decay):
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def sgd(model, lr, momentum, nesterov, weight_decay):
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)



def retrieve_optimizer(config, model):
    
    optim_name = config.NAME.lower()
    optimizer = None
    
    if optim_name == 'adam':
        optimizer = adam(model, config.LEARNING_RATE, config.WEIGHT_DECAY)
    elif optim_name == 'sgd':
        optimizer = sgd(model, config.LEARNING_RATE, config.MOMENTUM, config.NESTEROV, config.WEIGHT_DECAY)
    
    else:
        print("ERROR: Invalid optimizer configuration...")
        
    return optimizer
    
    