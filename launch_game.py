"""
Launching a game and playing against computer
"""

import logging

from games.tictactoe import Game
from mcts import MonteCarloTreeSearch


def main():
    """
    Run an interactive game with MCTS advice
    """
    logging.basicConfig(level=logging.CRITICAL)
    game = Game()
    # print init version of game
    game.show_board()
    while game.legal_plays():
        # run Monte Carlo Tree Search and show recommended move
        tree = MonteCarloTreeSearch(game=game)
        tree.search(max_iterations=10000, max_runtime=3, n_simulations=1)
        tree.show_tree(level=1)
        print("MCTS recommends: {}".format(tree.recommended_play()))
        # ask user for move to play and play it
        move = tuple([int(s) for s in input("Move to play (format: .,.) : ").split(',')])
        game.play(move)
        print('You played:')
        game.show_board()
        # a random  move is selected for opponent
        if game.legal_plays():
            game.play()
        print('Opponent played:')
        game.show_board()
    if game.winner() is None:
        print("It's a tie! :|")
    elif game.winner() == game.players[0]:
        print("You win! :)")
    elif game.winner() == game.players[1]:
        print("You lose! :(")
    else:
        raise ValueError('Unknown game status')


if __name__ == "__main__":
    main()
