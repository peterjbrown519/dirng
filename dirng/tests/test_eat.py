import unittest
from dirng.devices import Devices
from dirng.protocol import Protocol
from dirng.games import Game
from math import sqrt
from copy import deepcopy
import numpy as np

class TestEatOptimisation(unittest.TestCase):

    # setup
    def setUp(self):
        self.dev = Devices()
        chsh = Game('chsh', score = 0.75 + np.random.rand()*0.0135, delta = 1e-3,
                        matrix = [[0.25,0.00,0.25,0.00,0.00,0.00],
                                  [0.00,0.25,0.00,0.25,0.00,0.00],
                                  [0.25,0.00,0.00,0.25,0.00,0.00],
                                  [0.00,0.25,0.25,0.00,0.00,0.00]])
        align = Game('align', score = 0.8 + np.random.rand()*0.2, delta = 1e-3,
                        matrix = [[0.00,0.00,0.00,0.00,1.00,0.00],
                                  [0.00,0.00,0.00,0.00,0.00,1.00],
                                  [0.00,0.00,0.00,0.00,0.00,0.00],
                                  [0.00,0.00,0.00,0.00,0.00,0.00]])

        self.dev.generation_inputs = [1,2]
        self.dev.relaxation_level = 2
        self.dev.io_config = [[2,2], [2,2,2]]
        self.dev.games = [align, chsh]

        self.protocol = Protocol()
        self.protocol.n = 1e12
        self.protocol.y = 0.001

    def tearDown(self):
        self.dev = None
        self.protocol = None


    # defaults
    def test_optimisation(self):
        # Check entropy GA optimises
        eat_rate = self.protocol.eatRate(self.dev)
        new_rate, _ = self.protocol.eatRateGA(self.dev)
        self.assertLessEqual(eat_rate, new_rate)



if __name__ == '__main__':
    unittest.main()
