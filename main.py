import numpy as np
import yaml
import argparse
from easydict import EasyDict

# from pipeline import Pipeline
from adv_package import Logger
from adv_package.config import AttackManager, DefenseManager

import os


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = EasyDict(yaml.safe_load(file))
    
    return config


managerDict = {
    'ATTACK': AttackManager,
    'DEFENSE': DefenseManager
}


def main():
    parser = argparse.ArgumentParser(description='PyTorch Adversarial Attack package.')
    
    parser.add_argument('--config', required=True, type=str, help='Path to .yaml configuration file.')
    parser.add_argument('--gpus', default="0,1", type=str, help='GPU devices to use (0-7) (default: 0,1)')
    parser.add_argument('--debug', action='store_true')

    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    pipeline = managerDict[config.TYPE](config)

    loss, acc1, acc5 = pipeline.run_pipeline()

    if not args.debug:
        # Save history results
        if not os.path.exists(config.PATHS.RESULTS):
            os.makedirs(config.PATHS.RESULTS)

        np.save(config.PATHS.RESULTS + 'loss.npy', loss)
        np.save(config.PATHS.RESULTS + 'accuracy.npy', acc1)

        # Log best results
        logManager = Logger(config.LOGGER, config.TYPE, args.config)
        logManager.update((loss, acc1, acc5))

if __name__ == "__main__":
    main()