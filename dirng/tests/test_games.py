import unittest
import random, string
from dirng.games import Game

# Tests:
# 1. Defaults
# 2. Setting and getting
# 3. Score and delta boundaries

class TestGames(unittest.TestCase):

    # setup
    def setUp(self):
        self.game = Game()

    # defaults
    def test_defaults(self):
        self.assertEqual(self.game.name, None)
        self.assertEqual(self.game.matrix, None)
        self.assertEqual(self.game.score, 0.0)
        self.assertEqual(self.game.delta, 0.0)

    # setting/getting
    def test_setget(self):
        self.game.name = 'Scroopy Noopers'
        self.assertEqual(self.game.name, 'Scroopy Noopers')

        self.game.matrix = [[2,2],[2,2]]
        self.assertEqual(self.game.matrix, [[2,2],[2,2]])

        self.game.score = 2.0
        self.assertEqual(self.game.score, 2.0)

        self.game.delta = 0.2
        self.assertEqual(self.game.delta, 0.2)

    # sorting
    def test_sorting(self):
        random_strings = []
        games = []
        for _ in range(10):
            random_strings.append(''.join([random.choice(string.ascii_letters + string.digits) for n in range(8)]))
            games.append(Game(name = random_strings[-1]))

        self.assertLess(Game(name='Aardvark'), Game(name='Zebra'))
        self.assertEqual(Game(name = 'Horse'), Game(name = 'Horse'))

        game_names_sorted = [game.name for game in sorted(games)]
        self.assertEqual(game_names_sorted, sorted(random_strings))

if __name__ == '__main__':
    unittest.main()
