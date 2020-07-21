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


class GradientAttackTester(unittest.TestCase):
    '''
    Tests on GradientAttack class given configuration
    data for the 'ATTACK' and 'ADV_MODEL'
    '''

    @classmethod
    def setUpClass(cls) -> None:
        cls._config = load_config('tests/attack.test.yaml')
        # cls._att_ifgsm = attack.IFGSM(config.ADV_MODEL, config.ATTACK)

    def testEpsilon(self):
        config = copy.deepcopy(self._config)
        eps_type = {
            'RANDOM': attack.RandomEpsilon,
            'FIXED': attack.FixedEpsilon
        }
        for (eps, cls) in eps_type.items():
            with self.subTest(eps=eps):
                config.ATTACK.EPSILON.TYPE = eps
                c = attack.IFGSM(config.ADV_MODEL, config.ATTACK)
                self.assertIsInstance(c.epsManager, cls)
        del config

    def testEpsilonInvalidInput(self):
        ''' Handles unregistered or missing ATTACK.INIT parameter.'''
        config = copy.deepcopy(self._config)
        config.ATTACK.EPSILON.TYPE = 'ERROR'
        with self.assertRaises(KeyError):
            attack.IFGSM(config.ADV_MODEL, config.ATTACK)

        del config.ATTACK.EPSILON.TYPE

        with self.assertRaises(AttributeError):
            attack.IFGSM(config.ADV_MODEL, config.ATTACK)
        del config

    def testAlpha(self):
        ''' Ensures correct Alpha subclass is called based on parameter configuration.'''
        config = copy.deepcopy(self._config)
        alpha_type = {
            'DIVISOR': attack.DivisorAlpha,
        }

        for (alpha, cls) in alpha_type.items():
            with self.subTest(alpha=alpha):
                config.ATTACK.ALPHA.TYPE = alpha
                c = attack.IFGSM(config.ADV_MODEL, config.ATTACK)
                self.assertIsInstance(c.alphaManager, cls)
        del config

    def testInitValidInput(self):
        ''' Ensures correct Init subclass is called based on parameter configuration'''
        config = copy.deepcopy(self._config)
        init_type = {
            'RAW': attack.ZeroShift,
            'SHIFT': attack.RandomShift,
        }
        for (init, cls) in init_type.items():
            with self.subTest(type=init):
                config.ATTACK.INIT.TYPE = init
                c = attack.IFGSM(config.ADV_MODEL, config.ATTACK)
                self.assertIsInstance(c.initManager, cls)
        del config

    def testInitInvalidInput(self):
        ''' Handles unregistered or missing ATTACK.INIT parameter.'''
        config = copy.deepcopy(self._config)
        config.ATTACK.INIT.TYPE = 'ERROR'
        with self.assertRaises(KeyError):
            attack.IFGSM(config.ADV_MODEL, config.ATTACK)

        del config.ATTACK.INIT.TYPE

        with self.assertRaises(AttributeError):
            attack.IFGSM(config.ADV_MODEL, config.ATTACK)
        del config


    def testRun(self):
        ''' Test adversarial example generation. Constraint to a batch_size of 1.'''
        # 1. Hardcoded adversarial example (torch tensor format)
        x = torch.load('raw_image_ts.pt').unsqueeze(0)
        y = torch.cuda.LongTensor([3])

        iter2adv = {
            1: torch.load('adv_image_ts.pt').unsqueeze(0),
            4: torch.load('adv4_image_ts.pt').unsqueeze(0),
            10: torch.load('adv10_image_ts.pt').unsqueeze(0)
        }

        for (n_iter, adv) in iter2adv.items():
            with self.subTest(n_iter=n_iter):
                config = copy.deepcopy(self._config)
                config.ATTACK.MAX_ITER = n_iter
                config.ATTACK.ALPHA.DIVISOR = n_iter
                c = attack.IFGSM(config.ADV_MODEL, config.ATTACK)
                # Tensor Equality
                self.assertTrue(torch.all(adv.eq(c.run(x, y))))


    def testClampImage(self):
        ''' Ensures input is clamped within [0-1] range.'''
        x_input = torch.rand((64,3,32,32)) * 3. - 1
        x_input[0,0,0,0] = 2.3
        x_input[0,0,0,1] = -0.5
        self.assertFalse(torch.all(x_input.ge(torch.zeros((64, 3, 32, 32)))))
        self.assertFalse(torch.all(x_input.le(torch.ones((64, 3, 32, 32)))))
        x_clamp = attack.Attack.clampValidImage(x_input)
        self.assertTrue(torch.all(x_clamp.ge(torch.zeros((64, 3, 32, 32)))))
        self.assertTrue(torch.all(x_clamp.le(torch.ones((64,3,32,32)))))
        self.assertTrue(torch.all(x_input.ge(torch.zeros((64, 3, 32, 32)))))
        self.assertTrue(torch.all(x_input.le(torch.ones((64, 3, 32, 32)))))


    @classmethod
    def tearDownClass(cls) -> None:
        pass
        # del cls._attack



if __name__ == '__main__':
    unittest.main()
