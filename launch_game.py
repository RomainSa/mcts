"""
Launching a game and playing against computer
"""

import logging
from tictactoe import TicTacToe as Board
from mcts import MonteCarloTreeSearch


def main():
    """
    Run an interactive game with MCTS advice
    """
    logging.basicConfig(level=logging.CRITICAL)
    board_size = 3
    board = Board(size=board_size)
    # print init version of board
    board.display()
    while board.legal_plays():
        # run Monte Carlo Tree Search and show recommended move
        tree = MonteCarloTreeSearch(board_initialisation=board, node_init_params={'n_plays': 0, 'n_wins': 0})
        tree.search(max_iterations=1000, max_runtime=4, n_simulations=1)
        print("MCTS recommends: {}".format(tree.recommended_play()))
        # ask user for move to play and play it
        move = tuple([int(s) for s in input("Move to play (format: .,.) : ").split(',')])
        board.play(move)
        print('You played:')
        board.display()
        # a random  move is selected for opponent
        if board.legal_plays():
            board.play()
        print('Opponent played:')
        board.display()
    if board.winner() == 1:
        print("You win! :)")
    elif board.winner() == -1:
        print("You lose! :(")
    else:
        print("It's a tie! :|")


if __name__ == "__main__":
    main()
