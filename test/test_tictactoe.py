import unittest

import numpy as np
from games.tictactoe import Game


class TestTicTacToeMethods(unittest.TestCase):

    def setUp(self):
        self.board = Game(board_size=3, save_history=True)
        self.board.play(move=(0, 0))
        self.board.play(move=(1, 1))
        self.board.play(move=(0, 1))

    def test_display(self):
        displayed_board = 'O|O|.\n.|X|.\n.|.|.'
        self.assertEqual(self.board.show_board(return_string=True), displayed_board)

    def test_state(self):
        state = np.array([[1, 1, 0], [0, -1, 0], [0, 0, 0]])
        np.testing.assert_array_equal(self.board.state, state)

    def test_history(self):
        state0 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        np.testing.assert_array_equal(self.board.history[0], state0)
        state1 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        np.testing.assert_array_equal(self.board.history[1], state1)
        state2 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
        np.testing.assert_array_equal(self.board.history[2], state2)
        state3 = np.array([[1, 1, 0], [0, -1, 0], [0, 0, 0]])
        np.testing.assert_array_equal(self.board.history[3], state3)

    def test_legal_plays(self):
        true_legal_plays = {(0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)}
        legal_plays = set(self.board.legal_plays())
        self.assertEqual(legal_plays, true_legal_plays)

    def test_play(self):
        move = (2, 2)
        self.board.play(move=move)
        self.assertEqual(self.board.state[move], -1)

    def test_winner(self):
        move = (2, 2)
        self.board.play(move=move)
        move = (0, 2)
        self.board.play(move=move)
        self.assertEqual(self.board.winner().value, 1)

    def test_tie(self):
        for move in [(0, 2), (2, 0), (1, 0), (1, 2), (2, 1), (2, 2)]:
            self.board.play(move=move)
        self.assertEqual(self.board.winner(), None)


if __name__ == '__main__':
    unittest.main()
