import unittest
from dirng.devices import Devices
from dirng.games import Game
from math import sqrt
from copy import deepcopy
import numpy as np
# Tests:
# 1. Defaults
# 2. Setting and getting
# 3. Score and delta boundaries

class TestDevices(unittest.TestCase):

    # setup
    def setUp(self):
        self.dev = Devices({})
        self.chsh = Game('chsh', score = 0.85, delta = 1e-4,
                        matrix = [[0.25,0.00,0.25,0.00,0.00,0.00],
                                  [0.00,0.25,0.00,0.25,0.00,0.00],
                                  [0.25,0.00,0.00,0.25,0.00,0.00],
                                  [0.00,0.25,0.25,0.00,0.00,0.00]])
        self.align = Game('align', score = 0.99, delta = 1e-3,
                        matrix = [[0.00,0.00,0.00,0.00,1.00,0.00],
                                  [0.00,0.00,0.00,0.00,0.00,1.00],
                                  [0.00,0.00,0.00,0.00,0.00,0.00],
                                  [0.00,0.00,0.00,0.00,0.00,0.00]])

    def tearDown(self):
        self.dev = None
        self.chsh = None
        self.align = None


    # defaults
    def test_defaults(self):
        self.assertEqual(self.dev.name, 'device1')
        self.assertEqual(self.dev.io_config, [[2,2],[2,2]])
        self.assertEqual(self.dev.generation_inputs, [0,0])
        self.assertEqual(self.dev.relaxation_level, 2)
        self.assertEqual(self.dev.games, [])
        self.assertEqual(self.dev.score, [])
        self.assertEqual(self.dev.delta, [])
        self.assertEqual(self.dev.verbose, 0)

    # setting/getting
    def test_setviadict(self):
        settings = {'name' : 'Devina',
                    'io_config' : [[2,2], [2,2,2]],
                    'generation_inputs' : [1,2],
                    'relaxation_level' : 2,
                    'games' : [self.chsh, self.align],
                    'verbose' : 0}
        # Settings via dictionary
        dev = Devices(settings)
        self.assertEqual(dev.name, 'Devina')
        self.assertEqual(dev.io_config, [[2,2],[2,2,2]])
        self.assertEqual(dev.generation_inputs, [1,2])
        self.assertEqual(dev.games, [self.align,self.chsh])
        self.assertEqual(dev.score, [0.99,0.85])
        self.assertEqual(dev.delta, [0.001, 0.0001])
        self.assertEqual(dev.verbose, 0)

    def test_setviaattr(self):
        # Setting via attributes
        self.dev.name = 'Devina'
        self.dev.io_config = [[2,2],[2,2,2]]
        self.dev.generation_inputs = [1,2]
        self.dev.games += [self.chsh]
        self.dev.games += self.align
        self.dev.score = [0.9, 0.8]
        self.dev.delta = [0.001, 0.02]
        self.dev.verbose = 2
        self.assertEqual(self.dev.name, 'Devina')
        self.assertEqual(self.dev.io_config, [[2,2],[2,2,2]])
        self.assertEqual(self.dev.generation_inputs, [1,2])
        self.assertEqual(self.dev.games, [self.align, self.chsh])
        self.assertEqual(self.dev.score, [0.9,0.8])
        self.assertEqual(self.dev.delta, [0.001,0.02])
        self.assertEqual(self.dev.verbose, 2)

    def test_constraints(self):
        # Setting as before
        self.dev.name = 'Devina'
        self.dev.io_config = [[2,2],[2,2,2]]
        self.dev.generation_inputs = [1,2]
        self.dev.games += [self.chsh]
        self.dev.games += self.align
        self.dev.score = [0.9, 0.8]
        self.dev.delta = [0.001, 0.02]
        self.dev.verbose = 2

        with self.assertRaises(TypeError) as context:
            self.dev.name = 2
        with self.assertRaises(TypeError) as context:
            self.dev.relaxation_level = 2.0
        with self.assertRaises(OSError) as context:
            self.dev.solver = '/not/a/real/path/'
        with self.assertRaises(OSError) as context:
            self.dev.solver = '/scratch/pjb519/no_sdpa_here/'
        with self.assertRaises(TypeError) as context:
            self.dev.games = 'chsh'
        with self.assertRaises(ValueError) as context:
            self.dev.score = [1.0,2.0,3.0]
        with self.assertRaises(TypeError) as context:
            self.dev.score = 'hi'
        with self.assertRaises(ValueError) as context:
            self.dev.delta = [1.0,2.0,3.0]
        with self.assertRaises(TypeError) as context:
            self.dev.delta = 'hi'
        with self.assertRaises(TypeError) as context:
            self.dev.verbose = 'hi'

        self.dev.solver = '/scratch/pjb519/sdpa_here/'
        self.assertEqual(self.dev._solver_name, 'sdpa')
        self.dev.solver = '/scratch/pjb519/sdpa_here'
        self.assertEqual(self.dev._solver_name, 'sdpa')

    def test_solution(self):
        self.dev.io_config = [[2,2],[2,2,2]]
        self.dev.generation_inputs = [1,2]
        self.dev.relaxation_level = 2
        self.dev.games = [self.chsh, self.align]
        # Actual value should be around 1.299
        self.assertAlmostEqual(self.dev.hmin, 1.3, delta = 0.02)

    def test_functions(self):
        self.dev.io_config = [[2,2],[2,2,2]]
        self.dev.generation_inputs = [1,2]
        self.dev.relaxation_level = 2
        self.dev.games = [self.chsh, self.align]
        x = 0.25 + sqrt(2)/8
        distribution = [[x, 1/2-x, x, 1/2-x, 0.5, 0.0],
                        [1/2-x, x, 1/2-x, x, 0.0, 0.5],
                        [x, 1/2-x, 1/2-x, x, 0.25, 0.25],
                        [1/2-x, x, x, 1/2-x, 0.25, 0.25]]

        w = self.dev.distribution2Score(distribution)
        self.assertAlmostEqual(w[0], 1.0)
        self.assertAlmostEqual(w[1], 2*x)

        self.dev.score = w
        self.assertAlmostEqual(self.dev.hmin,2.0, delta = 0.01)

    def test_dualfuncisdualsol(self):
        self.dev.io_config = [[2,2],[2,2,2]]
        self.dev.generation_inputs = [1,2]
        self.dev.relaxation_level = 2
        self.dev.games = [self.chsh, self.align]
        temp_dev = deepcopy(self.dev)
        for k in range(20):
            chsh_score =  0.75 + np.random.rand()*0.1035
            align_score = 0.8 + 0.2*np.random.rand()
            av, lv, _, _v, status = temp_dev.dualSolution([align_score, chsh_score])
            self.dev.score = [align_score, chsh_score]
            self.assertAlmostEqual(av + np.dot(lv, _v), self.dev.gp, delta = 0.0001)



if __name__ == '__main__':
    unittest.main()
