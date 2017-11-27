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
        self.history = [self.state.copy()]   # copy() needed to avoid appending a reference
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
        if self.winner() == self.no_player_value:
            return list(zip(*np.where(self.state == self.no_player_value)))
        else:
            return []

    def winner(self):
        """
        Return the winner's value and display. If gale is tied, return no_player's value and display

        :return: tuple (winner value, winner display value)
        """
        # compute horizontal, vertical and diagonal sums
        sums = np.concatenate((np.sum(self.state, axis=0), np.sum(self.state, axis=1),
                               np.array([np.sum(np.diag(self.state)), np.sum(np.diag(self.state[::-1]))])))
        # check if sums correspond to a win and display winner
        if len(np.where(sums == self.size * self.player1_value)[0]) > 0:
            return self.player1_value
        elif len(np.where(sums == self.size * self.player2_value)[0]) > 0:
            return self.player2_value
        else:
            return self.no_player_value

    def display(self, state_number=-1):
        """
        Display the game board

        :return: nothing
        """
        no_player_display = '.'
        player1_display = 'O'
        player2_display = 'X'
        print('\n'.join(['|'.join(player1_display if e == self.player1_value else
                                  (player2_display if e == self.player2_value else no_player_display)
                                  for e in l) for l in self.history[state_number]]))

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
            self.history.append(self.state.copy())   # copy() needed to avoid appending a reference
        else:
            self.history = [self.state.copy()]   # only the current state is save (to be able to display it)
        self.current_player_value *= -1


if __name__ == "__main__":
    # plays a game and displays the board at each move
    board = TicTacToe(size=3, save_history=True)
    board.display()
    round = 0
    while len(board.legal_plays()) > 0:
        board.play()
        print('-'*6)
        round += 1
        print("ROUND NUMBER {}".format(round))
        board.display()
    print("WINNER: {}".format(board.winner()))
