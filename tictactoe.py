"""
Tic Tac Toe game implementation
"""

import numpy as np


class TicTacToe:
    """
    TiCTacToe game implementation to be used by Monte Carlo Tree Search
    https://en.wikipedia.org/wiki/Tic-tac-toe
    """

    def __init__(self, size=3, save_history=True):
        # board parameters and data
        self.size = size
        self.state = np.zeros((size, size), dtype=int)
        self.save_history = save_history
        self.history = [self.state.copy()]  # copy() needed to avoid appending a reference
        self.last_play = None
        # players parameters
        self.no_player_value = 0
        self.player1_value = 1
        self.player2_value = -1
        # player 1 is starting
        self.current_player_value = self.player1_value

    def legal_plays(self):
        """
        Takes a sequence of game states representing the full game history

        :return: the list of moves (as tuples) that are legal to play for the current player
        """
        plays = []
        if self.winner() == self.no_player_value:
            plays = list(zip(*np.where(self.state == self.no_player_value)))
        return plays

    def winner(self):
        """
        Return the winner's value and display. If gale is tied, return no_player's value and display

        :return: tuple (winner value, winner display value)
        """
        # compute horizontal, vertical and diagonal sums
        sums = np.concatenate((np.sum(self.state, axis=0), np.sum(self.state, axis=1),
                               np.array([np.sum(np.diag(self.state)), np.sum(np.diag(self.state[::-1]))])))
        # check if sums correspond to a win and display winner
        if np.where(sums == self.size * self.player1_value)[0].size > 0:
            winner_value = self.player1_value
        elif np.where(sums == self.size * self.player2_value)[0].size > 0:
            winner_value = self.player2_value
        else:
            winner_value = self.no_player_value
        return winner_value

    def display(self, state_number=-1, return_string=False):
        """
        Display the game board

        :return: nothing
        """
        no_player_display = '.'
        player1_display = 'O'
        player2_display = 'X'
        lines = []
        for line in self.history[state_number]:
            elements = []
            for element in line:
                if element == self.player1_value:
                    elements.append(player1_display)
                elif element == self.player2_value:
                    elements.append(player2_display)
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
        Plays one of the legal plays randomly

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
        # updates states and players info
        self.state[selected_move] = self.current_player_value
        if self.save_history:
            self.history.append(self.state.copy())  # copy() needed to avoid appending a reference
        else:
            self.history = [self.state.copy()]  # only the current state is save (to be able to display it)
        self.current_player_value *= -1
        self.last_play = selected_move


def main():
    """
    Run a TicTacToe game
    """
    # plays a game and displays the board at each move
    board = TicTacToe(size=3, save_history=True)
    board.display()
    n_round = 0
    while board.legal_plays():
        board.play()
        print('-' * 6)
        n_round += 1
        print("ROUND NUMBER {}".format(round))
        board.display()
    print("WINNER: {}".format(board.winner()))


if __name__ == "__main__":
    main()
