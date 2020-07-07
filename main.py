import numpy as np
import yaml
import argparse
from easydict import EasyDict

# from pipeline import Pipeline
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
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    pipeline = managerDict[config.TYPE](config)

    loss, accuracy = pipeline.run_pipeline()
    # loss, accuracy = pipeline.run_attack_test()
    
    if not os.path.exists(config.PATHS.RESULTS_PATH):
        os.makedirs(config.PATHS.RESULTS_PATH)
        
    np.save(config.PATHS.RESULTS_PATH + 'loss.npy', loss)
    np.save(config.PATHS.RESULTS_PATH + 'accuracy.npy', accuracy)
    
    
    

if __name__ == "__main__":
    main()