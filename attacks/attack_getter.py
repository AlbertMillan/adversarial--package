from .pgd import PGD

import sys


def retrieve(config, model, is_normalized):
    
    if config.NAME == 'PGD':
        return PGD(model, config.EPSILON, config.ALPHA, config.MAX_ITER, config.MOMENTUM, is_normalized)
    
    else:
        print("ERROR: Model name", config.NAME, "does not exist. Exiting...")
        sys.exit(1)