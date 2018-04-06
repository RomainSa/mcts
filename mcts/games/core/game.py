"""
Abstract class for a two players game
"""

from itertools import cycle

import numpy as np

from mcts.games.core.board import Board
from mcts.games.core.player import Player


class Game:
    """
    Abstract class for generic game
    """

    def __init__(self, board, players, save_history=True):
        """
        Game constructor

        :param board: a game board
        :type board: Board
        :param players: an iterable of game players
        :type players: List(Player)
        :param save_history: whether to save game history (memory-intensive) or not
        :type save_history: bool
        """

        # board input must be a Board object
        assert isinstance(board, Board)
        self.board = board

        # players input must be an iterable of Board objects
        for player in players:
            assert isinstance(player, Player)

        # players names, values and displays must be unique
        names = set()
        values = set()
        displays = set()

        for player in players:
            names.add(player.name)
            values.add(player.value)
            displays.add(player.display)

        assert len(players) == len(names)
        assert len(players) == len(values)
        assert len(players) == len(displays)
        self.players = players

        # history management
        self.save_history = save_history
        self._history = [self.board.state.copy()] if save_history else None  # copy() needed to avoid appending a ref

    def __str__(self):
        raise NotImplementedError()

    def legal_moves(self):
        raise NotImplementedError()

    def winner(self):
        raise NotImplementedError()

    def play(self):
        raise NotImplementedError()

    def run(self):
        raise NotImplementedError()


class TwoPlayersGame(Game):
    """
    Abstract class for generic two players game
    """

    def __init__(self, init_state, save_history=True):

        # prepare game and board
        players = [Player(name='player1', value=1, display='O'), Player(name='player2', value=-1, display='X')]
        board = Board(state=init_state, dtype=np.int8)
        Game.__init__(self, board, players, save_history)

        # save player turn to play
        self.last_player = None if init_state is None else self._get_last_player()
        if self.last_player is None or self.last_player == self.players[1]:
            self.players_generator = cycle(self.players)
        else:
            self.players_generator = cycle(self.players[::-1])
        self.current_player = next(self.players_generator)

    def __str__(self):
        raise NotImplementedError()

    def _get_last_player(self):
        """
        Calculates information about the last (and next) player.
         To be called only in case an init state is provided
        """
        # get counts per value (ie per player) and filter out no play (value is 0)
        values, counts = np.unique(self.board.state, return_counts=True)
        counts = [count for value, count in zip(values, counts) if value != 0]
        values = [value for value in values if value != 0]

        if len(values) == 0:
            # nobody has played yet
            return None

        elif len(values) == 1:
            # first player has played exactly once
            if values[0] == self.players[0].value and counts[0] == 1:
                return self.players[0]
            else:
                raise ValueError('One player has played more than once while the other has not played yet')

        elif len(values) == 2:
            # both players have played
            counts1 = counts[values.index(self.players[0].value)]    # plays of player 1
            counts2 = counts[values.index(self.players[1].value)]    # plays of player 2

            if counts1 - counts2 == 1:
                # last player is player 1 (one move in advance)
                return self.players[0]
            elif counts1 == counts2:
                # last player is player 2 (same number of moves)
                return self.players[1]
            else:
                raise ValueError('There are more than one play of difference between the two players')

        else:
            raise ValueError('There are more than two players')

    def legal_moves(self):
        raise NotImplementedError()

    def winner(self):
        raise NotImplementedError()

    def play(self):
        raise NotImplementedError()

    def run(self):
        raise NotImplementedError()
