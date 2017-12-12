import unittest

import numpy as np
from utils.scoring import average_wins, ucb1


class TestScoringMethods(unittest.TestCase):

    def test_average_wins(self):
        n_plays = 100
        n_ties = 10
        n_wins = 33
        score = average_wins(n_plays, n_wins, n_ties)

        self.assertAlmostEqual(score, 0.33, places=4)

    def test_ucb1(self):
        total_plays = 100
        n_plays = 50
        n_ties = 20
        n_wins = 30
        score = ucb1(n_plays, n_wins, n_ties, total_plays, c_=0.5)

        self.assertAlmostEqual(score, 0.7517, places=4)


if __name__ == '__main__':
    unittest.main()
