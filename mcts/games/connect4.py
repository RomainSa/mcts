"""
Connect4 game implementation
cf https://en.wikipedia.org/wiki/Connect_Four
"""

import logging
from itertools import cycle

import numpy as np

from mcts.games.core.board import Board
from mcts.games.core.game import TwoPlayersGame


class Connect4(TwoPlayersGame):

    def __init__(self, init_state=None, save_history=True):
        """
        :param init_state: initial board state
        :type init_state: np.array
        :param save_history: whether to save game history (memory-intensive) or not
        :type save_history: bool
        """

        # a Connect4 game has a 6x7 board
        if init_state is not None:
            assert init_state.shape == (6, 7)
            if init_state.dtype != np.dtype(np.int8):
                raise TypeError('Initial state must be of type: %s' % np.int8)
        board = Board(shape=(6, 7), state=init_state, dtype=np.int8)

        # call to Superclass constructor
        TwoPlayersGame.__init__(self, board, players, save_history)

        # game attributes
        self.board_size = (6, 7)

        # players attributes
        self.players_values = list([p.value for p in self.players])

        # player turn to play
        self.last_player = None if init_state is None else self._get_last_player()
        if self.last_player is None or self.last_player == self.players[1]:
            self.players_generator = cycle(self.players)
        else:
            self.players_generator = cycle(self.players[::-1])
        self.current_player = next(self.players_generator)

    def legal_plays(self):
        """
        Takes a sequence of game states representing the full game history
        Algorithms:

        :return: the list of moves tuples that are legal to play for the current player
        """
        legal_plays = []
        if self.winner() is None:
            free_spaces = np.isin(self.state, self.players_values, invert=True)
            legal_plays = np.argwhere(free_spaces.sum(axis=0)).ravel().tolist()
        logging.debug('Legal plays: %s', legal_plays)
        return legal_plays

    def winner(self):
        """
        Return the winner player. If game is tied, return None

        :return: Player or None
        """
        return self.winner_

    def show_board(self, state_number=-1, return_string=False):
        """
        Display the game game

        :param state_number: the state to show
        :param return_string: whether to return a string or to print it
        :return: game representation as a string or nothing
        """
        # creates the string representation of the game
        lines = []
        no_player_display = '.'
        for line in self.history[state_number]:
            elements = []
            for element in line:
                if element in self.players_values:
                    for player in self.players:
                        if element == player.value:
                            elements.append(player.display)
                else:
                    elements.append(no_player_display)
            lines.append('|'.join(elements))
        board_representation = '\n'.join(lines)

        if return_string:
            return board_representation
        else:
            print(board_representation)

    def play(self, move=None):
        """
        Play a move

        :param move: selected move to play (int corresponding to the column index)
        :return: nothing
        """
        legal_plays = self.legal_plays()

        if move is not None:
            # if input move is provided check that it is legal
            if move in legal_plays:
                selected_move = move
            else:
                raise ValueError('Selected move is illegal')
        else:
            # select a move randomly
            selected_move = legal_plays[np.random.choice(len(legal_plays), 1)[0]]
        logging.debug('Selected move: %s', move)

        # updates states
        row_number = self.board_size[0] - sum(self.state[:, selected_move] != 0) - 1
        self.state[row_number, selected_move] = self.current_player.value
        self.last_play = selected_move
        if self.save_history:
            self.history.append(self.state.copy())  # copy() needed to avoid appending a reference
        else:
            self.history = [self.state.copy()]  # only the current state is save (to be able to display it)

        # prepare data used to check for winner
        x = self.state
        i, j = row_number, selected_move
        pattern = ''.join([str(self.current_player.value)] * 4)
        spaces_to_check = [x[max(i-3, 0):i+4, j], x[i, max(j-3, 0):j+4],
                           np.diagonal(x, offset=j - i), np.diagonal(x[:, ::-1], offset=len(x)-(i+j))]

        # an ugly way to check if a list is a sublist
        for space in spaces_to_check:
            space_as_text = ''.join([str(i) for i in space])
            if pattern in space_as_text:
                self.winner_ = self.current_player
                return

        # updates player info
        self.current_player = next(self.players_gen)


def main():
    """
    Run a Connect4 game
    """
    # plays a game and displays the game at each move
    logging.basicConfig(level=logging.DEBUG)
    game = Game(board_size=(6, 7), save_history=True)
    game.show_board()
    n_round = 0
    while game.legal_plays():
        game.play()
        print('-' * 6)
        n_round += 1
        print("ROUND NUMBER {}".format(n_round))
        if game.winner() is not None:
            print("WINNER {}".format(game.winner().display))
        game.show_board()


if __name__ == "__main__":
    main()
