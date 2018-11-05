import unittest
from dirng.protocol import Protocol
from dirng.games import Game
from dirng.devices import Devices
# Tests:
# 1. Defaults
# 2. Setting and getting

class TestProtocol(unittest.TestCase):

    # setup
    def setUp(self):
        self.protocol = Protocol()

    def tearDown(self):
        self.protocol = None


    # defaults
    def test_defaults(self):
        self.assertEqual(self.protocol.n, 1e10)
        self.assertEqual(self.protocol.y, 1e-2)
        self.assertEqual(self.protocol.eps_smooth, 1e-8)
        self.assertEqual(self.protocol.eps_eat, 1e-8)

    # setting/getting
    def test_setviadict(self):
        settings = {'n' : 1e12,
                    'y' : 0.00012,
                    'eps_smooth' : 0.1234,
                    'eps_eat' : 0.99}
        # Settings via dictionary
        protocol = Protocol(settings)
        self.assertEqual(protocol.n, 1e12)
        self.assertEqual(protocol.y, 0.00012)
        self.assertEqual(protocol.eps_smooth, 0.1234)
        self.assertEqual(protocol.eps_eat, 0.99)

    def test_setviaattr(self):
        # Setting via attributes
        self.protocol.n = 123
        self.protocol.y = 1/2
        self.protocol.eps_smooth = 1/10
        self.protocol.eps_eat = 0.01
        self.assertEqual(self.protocol.n, 123)
        self.assertEqual(self.protocol.y, 0.5)
        self.assertEqual(self.protocol.eps_smooth, 0.1)
        self.assertEqual(self.protocol.eps_eat, 0.01)

    def test_constraints(self):
        # Check that assertions work
        with self.assertRaises(TypeError) as context:
            self.protocol.n = '1'
        with self.assertRaises(ValueError) as context:
            self.protocol.n = 0
        with self.assertRaises(TypeError) as context:
            self.protocol.y = 1
        with self.assertRaises(ValueError) as context:
            self.protocol.y = 0.0
        with self.assertRaises(ValueError) as context:
            self.protocol.y = 2.0
        with self.assertRaises(TypeError) as context:
            self.protocol.eps_smooth = 1
        with self.assertRaises(ValueError) as context:
            self.protocol.eps_smooth = 0.0
        with self.assertRaises(ValueError) as context:
            self.protocol.eps_smooth = -2.0
        with self.assertRaises(TypeError) as context:
            self.protocol.eps_eat = 1
        with self.assertRaises(ValueError) as context:
            self.protocol.eps_eat = 0.0
        with self.assertRaises(ValueError) as context:
            self.protocol.eps_eat = -2.0


    def test_fminStorage(self):
        dev = Devices()
        chsh = Game('chsh', score = 0.85, delta = 1e-4,
                        matrix = [[0.25,0.00,0.25,0.00,0.00,0.00],
                                  [0.00,0.25,0.00,0.25,0.00,0.00],
                                  [0.25,0.00,0.00,0.25,0.00,0.00],
                                  [0.00,0.25,0.25,0.00,0.00,0.00]])
        align = Game('align', score = 0.99, delta = 1e-3,
                        matrix = [[0.00,0.00,0.00,0.00,1.00,0.00],
                                  [0.00,0.00,0.00,0.00,0.00,1.00],
                                  [0.00,0.00,0.00,0.00,0.00,0.00],
                                  [0.00,0.00,0.00,0.00,0.00,0.00]])
        dev.io_config = [[2,2],[2,2,2]]
        dev.generation_inputs = [1,2]
        dev.relaxation_level = 2
        dev.games = [chsh, align]
        self.protocol.setFmin(dev)
        self.assertEqual(dev._fmin_variables[0], dev.dualSolution([0.99, 0.85])[0])
        self.assertSequenceEqual(dev._fmin_variables[1].tolist(), dev.dualSolution([0.99, 0.85])[1].tolist())
        self.assertSequenceEqual(dev._fmin_variables[2], dev.dualSolution([0.99, 0.85])[2])
        self.assertSequenceEqual(dev._fmin_variables[3].tolist(), dev.dualSolution([0.99, 0.85])[3].tolist())
        self.assertEqual(dev._fmin_variables[4], dev.dualSolution([0.99, 0.85])[4])

        # self.protocol.setFmin(dev, v_choice = [0.95, 0.851])
        # dev.score = [0.95, 0.851]
        # self.assertEqual(dev._fmin_variables[0], dev.dualSolution()[0])
        # self.assertListEqual(dev._fmin_variables[1], dev.dualSolution()[1])
        # self.assertSequenceEqual(dev._fmin_variables[2], dev.dualSolution()[2])
        # self.assertSequenceEqual(dev._fmin_variables[3], dev.dualSolution()[3])
        # self.assertEqual(dev._fmin_variables[4], dev.dualSolution()[4])

if __name__ == '__main__':
    unittest.main()
