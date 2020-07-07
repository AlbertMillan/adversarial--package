import unittest
from easydict import EasyDict
import yaml
import torch

from adv_package import attack

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = EasyDict(yaml.safe_load(file))

    return config

class GradientHelperTester(unittest.TestCase):

    def setUp(self) -> None:
        self._config = load_config('tests/helper.test.yaml')

    def testFixedEpsilon(self):
        ''' Test FixedEpsilon method. '''
        config = self._config
        c = attack.FixedEpsilon(config.EPSILON)

        eps_values = [0, 1, 4, 10]
        batch_size = [1, 10, 32, 64]

        for eps in eps_values:
            c.eps = eps
            for batch in batch_size:
                out = c.getEps(batch)
                exp = torch.full((batch,1,1,1), eps / 255.).cuda()
                self.assertEqual(out.size(), exp.size())
                self.assertTrue(torch.all(out.eq(exp)))

    def testRandomEpsilon(self):
        config = self._config
        c = attack.RandomEpsilon(config.EPSILON)

        batch_sizes = [1, 32, 64]

        for batch in batch_sizes:
            lower_bound = torch.full((batch,1,1,1), config.EPSILON.MIN_VAL).cuda() / 255.
            upper_bound = torch.full((batch,1,1,1), config.EPSILON.MAX_VAL).cuda() / 255.
            out = c.getEps(batch)
            self.assertTrue(torch.all(out.ge(lower_bound)))
            self.assertTrue(torch.all(out.le(upper_bound)))


    def testRandomShift(self):
        ''' Returns random positive negative values within epsilon l-inf ball. '''
        config = self._config

        eps_values = [0.39, .078, .0199]
        batch_sizes = [(1,3,32,32), (10,3,32,32), (64,3,32,32)]

        c = attack.RandomShift()

        for eps in eps_values:

            for batch in batch_sizes:
                lower_bound = torch.full(batch, -eps).cuda()
                upper_bound = torch.full(batch, eps).cuda()
                eps_ts = torch.full((batch[0], 1, 1, 1), eps).cuda()
                out = c.getShift(batch, eps_ts)
                self.assertTrue(torch.all(out.ge(lower_bound)) and torch.all(out.le(upper_bound)))

    def tearDown(self) -> None:
        del self._config

if __name__ == '__main__':
    unittest.main()
