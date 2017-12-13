import unittest

import numpy as np
from games.connect4 import Game


class TestConnect4Methods(unittest.TestCase):

    def setUp(self):
        self.game = Game(board_size=(6, 7), save_history=True)
        moves = [3, 1, 3, 2, 5, 4, 4, 2, 2, 3, 4]
        for move in moves:
            self.game.play(move=move)

    def test_display(self):
        displayed_board = '.|.|.|.|.|.|.\n.|.|.|.|.|.|.\n.|.|.|.|.|.|.\n.|.|O|X|O|.|.\n.|.|X|O|O|.|.\n.|X|X|O|X|O|.'
        self.assertEqual(self.game.show_board(return_string=True), displayed_board)

    def test_current_player(self):
        self.assertEqual(self.game.current_player, self.game.players[1])

    def test_history(self):
        state0 = np.zeros(self.game.board_size)
        np.testing.assert_array_equal(self.game.history[0], state0)
        state1 = state0.copy()
        state1[5, 3] = 1
        np.testing.assert_array_equal(self.game.history[1], state1)
        state2 = state1.copy()
        state2[5, 1] = 2
        np.testing.assert_array_equal(self.game.history[2], state2)

    def test_legal_plays1(self):
        true_legal_plays = set(range(self.game.board_size[1]))
        legal_plays = set(self.game.legal_plays())
        self.assertEqual(legal_plays, true_legal_plays)

    def test_legal_plays2(self):
        moves = [3, 3, 3, 4, 4, 4]
        for move in moves:
            self.game.play(move)
        true_legal_plays = {0, 1, 2, 5, 6}
        legal_plays = set(self.game.legal_plays())
        self.assertEqual(legal_plays, true_legal_plays)

    def test_play(self):
        move = 2
        player = self.game.current_player
        self.game.play(move=move)
        self.game.show_board()
        self.assertEqual(self.game.state[2, move], player.value)

    def test_winner(self):
        self.assertIsNone(self.game.winner())

        move = 4
        self.game.play(move=move)
        self.assertEqual(self.game.winner(), self.game.players[1])


if __name__ == '__main__':
    unittest.main()
