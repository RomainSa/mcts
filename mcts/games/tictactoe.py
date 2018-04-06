"""
Tic Tac Toe game implementation
cf https://en.wikipedia.org/wiki/Tic-tac-toe
"""

import logging
from itertools import cycle

import numpy as np

from mcts.games.core.board import Board
from mcts.games.core.game import TwoPlayersGame


class TicTacToe(TwoPlayersGame):

    def __init__(self, init_state=None, save_history=True):
        """
        :param init_state: initial board state
        :type init_state: np.array
        :param save_history: whether to save game history (memory-intensive) or not
        :type save_history: bool
        """
        # a TicTacToe game must have a 3x3 board
        if init_state is not None:
            assert init_state.shape == (3, 3)
            if init_state.dtype != np.dtype(np.int8):
                raise TypeError('Initial state must be of type: %s' % np.int8)
        board = Board(shape=(3, 3), state=init_state, dtype=np.int8)

        # call to Superclass constructor
        TwoPlayersGame.__init__(self, board, save_history)

        # players attributes
        self.players_values = list([p.value for p in self.players])

        # player turn to play
        self.last_player = None if init_state is None else self._get_last_player()
        if self.last_player is None or self.last_player == self.players[1]:
            self.players_generator = cycle(self.players)
        else:
            self.players_generator = cycle(self.players[::-1])
        self.current_player = next(self.players_generator)

        # internals
        self._sums = None   # to look for a winner
        self._compute_sums()
        self._players_dict = {player.value: player for player in players}   # to get a player from its value

    def __str__(self):
        """
        String representation of the game
        """
        # creates the string representation of the game
        lines = []
        no_player_display = '.'

        for line in self.board.state:
            elements = []
            for element in line:
                if element in self.players_values:
                    elements.append(self._players_dict[element].display)
                else:
                    elements.append(no_player_display)
            lines.append('|'.join(elements))
        board_representation = '\n'.join(lines)

        return board_representation

    def _compute_sums(self):
        """ Sums players values on board to look for a winner """
        # updates sums that are used to check for winner
        self._sums = np.concatenate(
            (np.sum(self.board.state, axis=0),   # vertical
             np.sum(self.board.state, axis=1),   # horizontal
             np.array([np.sum(np.diag(self.board.state)),   # diagonals
                       np.sum(np.diag(self.board.state[::-1]))])))

    def legal_moves(self):
        """
        Return all the moves that the current player can legally play

        :return: the list of moves tuples that are legal to play for the current player
        :rtype: list
        """
        legal_moves = []
        if self.winner() is None:
            free_spaces = np.isin(self.board.state, self.players_values, invert=True)
            legal_moves = np.argwhere(free_spaces)
            legal_moves = list(map(tuple, legal_moves))   # convert numpy array to list of tuples
        logging.debug('Legal moves: %s', legal_moves)
        return legal_moves

    def winner(self):
        """
        Return the winner player. If game is tied, return None

        :return: Player or None
        """
        for player in self.players:
            if len(self.board.state) * player.value in self._sums:   # one axis is full of this player plays (= win)
                logging.debug('Winner: %s', player.display)
                return player
        logging.debug('Winner: None')
        return None   # no winner found

    def play(self, move=None):
        """
        Play a move

        :param move: selected move to play. If None it is chosen randomly among legal plays
        :return: Nothing
        """
        legal_plays = self.legal_moves()

        if move is not None:
            # if input move is provided check that it is legal
            if move in legal_plays:
                selected_move = move
            else:
                raise ValueError('Selected move is illegal')
        else:
            # select a move randomly
            selected_move = legal_plays[np.random.choice(len(legal_plays), 1)[0]]
        logging.debug('Selected move: %s', selected_move)

        # updates states and players info
        self.board.state[selected_move] = self.current_player.value
        if self.save_history:
            self._history.append(self.board.state.copy())  # copy() needed to avoid appending a reference
        self.current_player = next(self.players_generator)
        self.last_player = selected_move

        # updates sums that are used to check for winner
        self._compute_sums()

    def run(self):
        """ Run a TicTacToe game """
        # plays a game and displays the game at each move
        n_round = 0
        logging.debug('-' * 13)
        logging.debug('ROUND NUMBER %s', n_round)
        logging.debug('\n' + self.__str__())
        while self.legal_moves():
            self.play()
            logging.debug('-' * 13)
            n_round += 1
            logging.debug('ROUND NUMBER %s', n_round)
            logging.debug('\n' + self.__str__())
