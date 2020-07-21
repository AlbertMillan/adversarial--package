import unittest
from easydict import EasyDict
import yaml
import copy

from adv_package.utils import Logger

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = EasyDict(yaml.safe_load(file))

    return config


class UtilsTester(unittest.TestCase):

    def setUp(self) -> None:
        # Read config file
        self.config = load_config('tests/utils.test.yaml')

    def testSaver(self):
        cfg = copy.deepcopy(self.config)
        c = Logger(cfg.LOGGER, 'DEFENSE', 'tests/utils.test.yaml')
        c.update((0.1212, 0.3131, 0.5151))

        with open(cfg.LOGGER.FILE, 'r') as f:
            lines = f.read().splitlines()
            last_line = lines[-1]
            self.assertEqual(last_line, 'TYPE: DEFENSE \t CONFIG: tests/utils.test.yaml \t LOSS: 0.121 \t TOP1: 0.313 \t TOP5: 0.515')

    def testAccuracy(self):



if __name__ == '__main__':
    unittest.main()
