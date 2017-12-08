"""
Tic Tac Toe game implementation
"""

import logging
from itertools import cycle

import numpy as np


class Player:
    """
    TicTacToe player
    """
    def __init__(self, name, value, display):
        self.name = name
        self.value = value
        self.display = display


class Board:
    """
    TicTacToe game implementation to be used by Monte Carlo Tree Search
    https://en.wikipedia.org/wiki/Tic-tac-toe
    """

    def __init__(self, size=3, save_history=True):
        # board
        self.size = size
        self.state = np.zeros((size, size), dtype=int)
        self.save_history = save_history
        self.history = [self.state.copy()]  # copy() needed to avoid appending a reference
        self.last_play = None
        # players
        self.players = [Player(name='A', value=1, display='O'), Player(name='B', value=-1, display='X')]
        # internal calculations
        self.__sums__ = np.array([])
        self.__players_values__ = list([p.value for p in self.players])
        self.__players_gen__ = cycle(self.players)
        self.current_player = next(self.__players_gen__)

    def legal_plays(self):
        """
        Takes a sequence of game states representing the full game history

        :return: the list of moves tuples that are legal to play for the current player
        """
        plays = []
        if self.winner() is None:
            plays = np.argwhere(np.isin(self.state, self.__players_values__, invert=True))   # free board spaces
            plays = list(map(tuple, plays))   # convert numpy array to list of tuples
        logging.debug('Legal plays: %s', plays)
        return plays

    def winner(self):
        """
        Return the winner player. If game is tied, return None

        :return: Player or None
        """
        for player in self.players:
            if self.size * player.value in self.__sums__:   # one axis is full of this player plays (= win)
                logging.debug('Winner: %s', player.display)
                return player
        logging.debug('Winner: None')
        return None   # no winner found

    def display(self, state_number=-1, return_string=False):
        """
        Display the game board

        :return: board representation as a string or nothing
        """
        # creates the string representation of the board
        lines = []
        no_player_display = '.'
        for line in self.history[state_number]:
            elements = []
            for element in line:
                if element in self.__players_values__:
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
        Play one of the legal plays randomly

        :return: nothing
        """
        # get all legal plays
        legal_plays = self.legal_plays()
        if move is not None:
            if move in legal_plays:
                selected_move = move
            else:
                raise ValueError('Selected move is illegal')
        else:
            # select a move randomly
            selected_move = legal_plays[np.random.choice(len(legal_plays), 1)[0]]
        logging.debug('Selected move: %s', move)
        # updates states and players info
        self.state[selected_move] = self.current_player.value
        if self.save_history:
            self.history.append(self.state.copy())  # copy() needed to avoid appending a reference
        else:
            self.history = [self.state.copy()]  # only the current state is save (to be able to display it)
        self.current_player = next(self.__players_gen__)
        self.last_play = selected_move
        # updates sums that are used to check for winner
        self.__sums__ = np.concatenate((np.sum(self.state, axis=0), np.sum(self.state, axis=1),
                                        np.array([np.sum(np.diag(self.state)), np.sum(np.diag(self.state[::-1]))])))


def main():
    """
    Run a TicTacToe game
    """
    # plays a game and displays the board at each move
    logging.basicConfig(level=logging.DEBUG)
    board = Board(size=3, save_history=True)
    board.display()
    n_round = 0
    while board.legal_plays():
        board.play()
        print('-' * 6)
        n_round += 1
        print("ROUND NUMBER {}".format(n_round))
        board.display()


if __name__ == "__main__":
    main()
