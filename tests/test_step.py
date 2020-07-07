import unittest
import torch
from easydict import EasyDict
import yaml
import copy

from adv_package import attack


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = EasyDict(yaml.safe_load(file))

    return config


class StepTester(unittest.TestCase):
    '''
    Tests on GradientAttack class given configuration
    data for the 'ATTACK' and 'ADV_MODEL'
    '''

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

if __name__ == '__main__':
    unittest.main()
