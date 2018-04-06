import unittest

import numpy as np
from mcts.games.core.game import TwoPlayersGame


class TestGames(unittest.TestCase):

    def test_game(self):
        pass

    def test_two_players_game(self):
        # valid init states
        init_state = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int8)
        game = TwoPlayersGame(init_state=init_state)
        assert game.last_player is None
        assert game.current_player == game.players[0]

        init_state = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.int8)
        game = TwoPlayersGame(init_state=init_state)
        assert game.last_player == game.players[0]
        assert game.current_player == game.players[1]

        init_state = np.array([[0, 0, 0], [0, 0, 1], [-1, 0, 0]], dtype=np.int8)
        game = TwoPlayersGame(init_state=init_state)
        assert game.last_player == game.players[1]
        assert game.current_player == game.players[0]

        init_state = np.array([[0, 0, 0], [0, 0, 1], [-1, 0, 1]], dtype=np.int8)
        game = TwoPlayersGame(init_state=init_state)
        assert game.last_player == game.players[0]
        assert game.current_player == game.players[1]

        # invalid init states
        init_state = np.array([[0, 0, 0], [0, 1, 1], [-1, 1, 1]], dtype=np.int8)
        with self.assertRaises(ValueError):
            TwoPlayersGame(init_state=init_state)

        init_state = np.array([[0, 0, 0], [0, 0, 0], [1, 2, 3]], dtype=np.int8)
        with self.assertRaises(ValueError):
            TwoPlayersGame(init_state=init_state)

        init_state = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]], dtype=np.int8)
        with self.assertRaises(ValueError):
            TwoPlayersGame(init_state=init_state)


if __name__ == '__main__':
    unittest.main()
