import unittest

import numpy as np
from mcts.games.tictactoe import TicTacToe


class TestTicTacToeMethodsWithHistory(unittest.TestCase):

    def setUp(self):
        # game with history
        self.game = TicTacToe(save_history=True)
        self.game.play(move=(0, 0))
        self.game.play(move=(1, 1))
        self.game.play(move=(0, 1))

        # game without history
        self.game_nohistory = TicTacToe(save_history=False)
        self.game_nohistory.play(move=(0, 0))
        self.game_nohistory.play(move=(1, 1))
        self.game_nohistory.play(move=(0, 1))

        # game with init state
        init_state = np.array([[1, 1, 0], [0, -1, 0], [0, 0, 0]], dtype=np.int8)
        self.game_init = TicTacToe(init_state=init_state)

    def test_init(self):
        with self.assertRaises(TypeError):
            TicTacToe(init_state=np.zeros((3, 3), dtype=np.float64))

    def test_display(self):
        self.assertEqual(self.game.__str__(), 'O|O|.\n.|X|.\n.|.|.')
        self.assertEqual(self.game_nohistory.__str__(), 'O|O|.\n.|X|.\n.|.|.')
        self.assertEqual(self.game_init.__str__(), 'O|O|.\n.|X|.\n.|.|.')

    def test_state(self):
        state = np.array([[1, 1, 0], [0, -1, 0], [0, 0, 0]])
        np.testing.assert_array_equal(self.game.board.state, state)
        np.testing.assert_array_equal(self.game_nohistory.board.state, state)
        np.testing.assert_array_equal(self.game_init.board.state, state)

    def test_history(self):
        state0 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int8)
        state1 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int8)
        state2 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=np.int8)
        state3 = np.array([[1, 1, 0], [0, -1, 0], [0, 0, 0]], dtype=np.int8)

        # no history
        self.assertIsNone(self.game_nohistory._history)

        # init state
        np.testing.assert_array_equal(self.game_init._history[0], state3)

        # empty game
        np.testing.assert_array_equal(self.game._history[0], state0)
        np.testing.assert_array_equal(self.game._history[1], state1)
        np.testing.assert_array_equal(self.game._history[2], state2)
        np.testing.assert_array_equal(self.game._history[3], state3)

    def test_legal_moves(self):
        true_legal_moves = {(0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)}
        self.assertEqual(set(self.game.legal_moves()), true_legal_moves)
        self.assertEqual(set(self.game_nohistory.legal_moves()), true_legal_moves)
        self.assertEqual(set(self.game_init.legal_moves()), true_legal_moves)

    def test_play(self):
        # valid move
        move = (2, 2)
        self.game.play(move=move)
        self.game_nohistory.play(move=move)
        self.game_init.play(move=move)
        self.assertEqual(self.game.board.state[move], -1)
        self.assertEqual(self.game_nohistory.board.state[move], -1)
        self.assertEqual(self.game_init.board.state[move], -1)

        # random move
        n_nonzeros = np.count_nonzero(self.game.board.state)
        self.game.play()
        self.game_nohistory.play()
        self.game_init.play()
        self.assertEqual(np.count_nonzero(self.game.board.state), n_nonzeros+1)
        self.assertEqual(np.count_nonzero(self.game_nohistory.board.state), n_nonzeros+1)
        self.assertEqual(np.count_nonzero(self.game_init.board.state), n_nonzeros+1)

        # invalid move
        move = (-1, -1)
        with self.assertRaises(ValueError):
            self.game.play(move=move)

    def test_winner(self):
        move = (2, 2)
        self.game.play(move=move)
        self.game_nohistory.play(move=move)
        self.game_init.play(move=move)
        move = (0, 2)
        self.game.play(move=move)
        self.game_nohistory.play(move=move)
        self.game_init.play(move=move)

        self.assertEqual(self.game.winner().value, 1)
        self.assertEqual(self.game_nohistory.winner().value, 1)
        self.assertEqual(self.game_init.winner().value, 1)

    def test_tie(self):
        for move in [(0, 2), (2, 0), (1, 0), (1, 2), (2, 1), (2, 2)]:
            self.game.play(move=move)
            self.game_nohistory.play(move=move)
            self.game_init.play(move=move)

        self.assertEqual(self.game.winner(), None)
        self.assertEqual(self.game_nohistory.winner(), None)
        self.assertEqual(self.game_init.winner(), None)


if __name__ == '__main__':
    unittest.main()
