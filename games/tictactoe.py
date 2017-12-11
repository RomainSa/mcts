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


class Game:
    """
    TicTacToe game implementation to be used by Monte Carlo Tree Search
    https://en.wikipedia.org/wiki/Tic-tac-toe
    """

    def __init__(self, board_size=3, save_history=True):
        # board attributes
        self.board_size = board_size
        self.state = np.zeros((board_size, board_size), dtype=int)
        self.save_history = save_history
        self.history = [self.state.copy()]  # copy() needed to avoid appending a reference
        self.last_play = None
        self.sums = np.array([])
        # players attributes
        self.players = [Player(name='A', value=1, display='O'), Player(name='B', value=-1, display='X')]
        self.players_values = list([p.value for p in self.players])
        self.players_gen = cycle(self.players)
        self.current_player = next(self.players_gen)

    def legal_plays(self):
        """
        Takes a sequence of game states representing the full game history

        :return: the list of moves tuples that are legal to play for the current player
        """
        legal_plays = []
        if self.winner() is None:
            free_spaces = np.isin(self.state, self.players_values, invert=True)
            legal_plays = np.argwhere(free_spaces)
            legal_plays = list(map(tuple, legal_plays))   # convert numpy array to list of tuples
        logging.debug('Legal plays: %s', legal_plays)
        return legal_plays

    def winner(self):
        """
        Return the winner player. If game is tied, return None

        :return: Player or None
        """
        for player in self.players:
            if self.board_size * player.value in self.sums:   # one axis is full of this player plays (= win)
                logging.debug('Winner: %s', player.display)
                return player
        logging.debug('Winner: None')
        return None   # no winner found

    def show_board(self, state_number=-1, return_string=False):
        """
        Display the game board

        :param state_number: the state to show
        :param return_string: whether to return a string or to print it
        :return: board representation as a string or nothing
        """
        # creates the string representation of the board
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

        :param move: selected move to play. If None it is chosen randomly amon legal plays
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

        # updates states and players info
        self.state[selected_move] = self.current_player.value
        if self.save_history:
            self.history.append(self.state.copy())  # copy() needed to avoid appending a reference
        else:
            self.history = [self.state.copy()]  # only the current state is save (to be able to display it)
        self.current_player = next(self.players_gen)
        self.last_play = selected_move

        # updates sums that are used to check for winner
        self.sums = np.concatenate(
            (np.sum(self.state, axis=0),   # vertical
             np.sum(self.state, axis=1),   # horizontal
             np.array([np.sum(np.diag(self.state)),   # diagonal
                       np.sum(np.diag(self.state[::-1]))])))


def main():
    """
    Run a TicTacToe game
    """
    # plays a game and displays the game at each move
    logging.basicConfig(level=logging.DEBUG)
    game = Game(board_size=3, save_history=True)
    game.show_board()
    n_round = 0
    while game.legal_plays():
        game.play()
        print('-' * 6)
        n_round += 1
        print("ROUND NUMBER {}".format(n_round))
        game.show_board()


if __name__ == "__main__":
    main()
