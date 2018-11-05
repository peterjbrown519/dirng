import unittest
import numpy as np
from dirng.devices import Devices
from dirng.games import Game
from dirng.qubit_methods import distribution, optimiseQubitGP, angles2Score, angles2GP
from math import pi
# Tests:
# 1. Defaults
# 2. Setting and getting
# 3. Score and delta boundaries

class TestQubits(unittest.TestCase):

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


    def tearDown(self):
        self.dev = None


    # defaults
    def test_angles2score(self):
        theta = pi/4
        a_ang = [0.0, pi/2]
        b_ang = [pi/4, -pi/4, 0.0]
        w = angles2Score(self.dev, [theta, a_ang, b_ang])
        self.assertAlmostEqual(w[0], 1.0, delta = 0.0001)
        self.assertAlmostEqual(w[1], 0.8535, delta = 0.0001)

    def test_optimisation(self):
        # Perturb about optimal and see if we can optimise back. 
        theta = np.random.normal(pi/4, pi/16)
        a_ang = [np.random.normal(0.0, pi/16),np.random.normal(pi/2, pi/16)]
        b_ang = [np.random.normal(pi/4, pi/16),np.random.normal(-pi/4, pi/16), np.random.normal(0.0, pi/16)]
        original_gp = angles2GP(self.dev, [theta, a_ang, b_ang])
        optimised_gp, _ = optimiseQubitGP(self.dev, [theta, a_ang, b_ang])
        self.assertLessEqual(optimised_gp, original_gp)
        self.assertAlmostEqual(optimised_gp, 0.25, delta=(original_gp-0.25)/2)



if __name__ == '__main__':
    unittest.main()
