"""
Monte Carlo Tree Search implementation
"""

import logging
import time
from copy import deepcopy

import numpy as np
from anytree import Node, LevelOrderGroupIter, RenderTree
from utils.scoring import ucb1, average_wins

from games.tictactoe import Game


class MonteCarloTreeSearch:
    """
    Implementation of the Monte Carlo Tree Search algorithm
    Based on http://mcts.ai/pubs/mcts-survey-master.pdf
    """

    def __init__(self, game):
        self.game = game
        self.node_init_params = {'n_plays': 0, 'n_wins': 0, 'n_ties': 0, 'score': 0.}
        self.root = Node('0', game=self.game, **self.node_init_params)

    def select(self, scoring_func=ucb1):
        """
        Select a node of the tree based on scores or expand current one if not all children have been visited

        :param scoring_func: the function that takes as inputs (n_plays, n_wins, n_ties) and output the node score
        :return: Node with best score
        """
        # selection start from root node
        node = self.root

        # browse each level until we reach a terminal node
        while node.children:
            if len(node.game.legal_plays()) > len(node.children):
                # if node still has unexplored children we select it
                logging.debug('-SELECT- chose node: %s that was not completely expanded', node.name)
                return node
            else:
                # we go down the tree until we reach the bottom always choosing the best score at each level
                nodes = node.children
                scores = []
                for node_ in nodes:
                    total_plays = node_.parent.n_plays
                    node_.score = scoring_func(plays=node_.n_plays,
                                               wins=node_.n_wins,
                                               ties=node_.n_ties,
                                               total_plays=total_plays)
                    scores.append(node_.score)
                node = nodes[np.argmax(scores)]
                logging.debug('-SELECT- chose temporary best score node: %s', node.name)
        logging.debug('-SELECT- final choice is node: %s', node.name)
        return node

    def expand(self, parent):
        """
        Randomly expand a child for selected node in order to expand the tree

        :param parent: Node to expand
        :return: Expanded child node
        """
        # filter out plays that already have been expanded
        already_played = [node.game.last_play for node in parent.children]
        unexplored_plays = [play for play in parent.game.legal_plays() if play not in already_played]
        if unexplored_plays:
            # choose one play randomly
            selected_play = unexplored_plays[np.random.choice(len(unexplored_plays), 1)[0]]
            # create a new node where this play is performed
            child_game = deepcopy(parent.game)
            child_game.play(selected_play)
            child_name = parent.name + '_' + str(len(parent.children))
            child = Node(name=child_name, parent=parent, game=child_game, **self.node_init_params)
            logging.debug('-EXPAND- played %s from node %s to new child %s', selected_play, parent.name, child.name)
            node = child
        else:
            # if all nodes have been explored return parent without expanding (can happen at end of tree search)
            logging.debug('-EXPAND- did not expanded parent %s', parent.name)
            node = parent
        return node

    def simulate(self, node, n_simulations):
        """
        Simulate games from current game state and returns number of wins

        :param node: node from which the simulated games start
        :param n_simulations: number of games simulations to perform
        :return: number of time the current player has won
        """
        n_wins = 0
        n_ties = 0
        for _ in range(n_simulations):
            # play until the end of the game
            game = deepcopy(node.game)
            node_player = game.current_player
            logging.debug('-SIMULATE- from state\n%s\n with player %s', game.show_board(return_string=True),
                          node_player.display)
            while game.legal_plays():
                game.play()
            # update win stats
            if game.winner() == node_player:
                n_wins += 1
            elif game.winner() is None:
                n_ties += 1
            logging.debug('-SIMULATE- and ended on state\n%s', game.show_board(return_string=True))
        logging.debug('-SIMULATE- performed %s plays and got %s wins', n_simulations, n_wins)
        return n_wins, n_ties

    def backpropagate(self, node, n_plays, n_wins, n_ties):
        """
        Back-propagate the results of the simulations to the ancestor nodes of the tree

        :param node: starting node for backpropagation (from bottom to top)
        :param n_plays: number of games played to backpropagate
        :param n_wins: number of games won to backpropagate
        :param n_ties: number of ties to backpropagate
        :return: nothing
        """
        # apply updates on current node
        node.n_plays += n_plays
        node.n_wins += n_wins
        node.n_ties += n_ties

        # and its ancestors
        for ancestor in node.ancestors:
            ancestor.n_plays += n_plays
            ancestor.n_ties += n_ties
            logging.debug('-BACKPROPAGATED- %s plays (of which %s ties) to ancestor node %s',
                          n_plays,
                          n_ties,
                          ancestor.name)
            # depending on the player the number of wins is not the same
            if node.game.current_player == ancestor.game.current_player:
                ancestor.n_wins += n_wins
                logging.debug('-BACKPROPAGATED- %s wins to same player ancestor node %s', n_wins, ancestor.name)
            else:
                ancestor.n_wins += (n_plays - n_ties - n_wins)
                logging.debug('-BACKPROPAGATED- %s wins to different player ancestor node %s', n_plays - n_wins,
                              ancestor.name)

    def show_tree(self, return_string=False, level=-1):
        """
        Print the current state of the tree along with some statistics on nodes

        :param return_string: boolean whether to return a string or to print the tree
        :param level: max level to print. If -1 print full tree
        :return: tree representation as a string or nothing if printed
        """
        def sort_by_move(nodes):
            """
            Sort nodes by move from (0, 0) to (2,2)

            :param nodes: list of nodes
            :return: sorted list
            """
            return sorted(nodes, key=lambda n: n.game.last_play)

        result = ['\n']
        output = '%s%s | Last move is %s | Player %s turn | %s wins and %s ties in %s plays | score: %.3f'

        nodes_selections = []
        if level > 0:
            # from list of tuples of nodes to list of nodes
            nodes_selections = [e for sub in list(LevelOrderGroupIter(self.root))[:level+1] for e in sub]

        for indent, _, node in RenderTree(self.root, childiter=sort_by_move):
            if level == -1 or node in nodes_selections:
                result.append((output % (indent,
                                         node.name,
                                         node.game.last_play,
                                         node.game.current_player.display,
                                         node.n_wins,
                                         node.n_ties,
                                         node.n_plays,
                                         node.score)))
        result = '\n'.join(result)
        if return_string:
            return result
        print(result)

    def search(self, max_iterations, max_runtime, n_simulations=1, display_tree=False):
        """
        Run a Monte Carlo Tree Search starting from root node

        :param max_iterations: max number of iterations for the tree search
        :param max_runtime: max search time in seconds
        :param n_simulations: number of simulations per expanded node
        :param display_tree: whether or not the tree is printed at each iteration
        """
        i, starting_time, ending_time = 0, time.time(), time.time() + max_runtime
        node = self.root
        while i < max_iterations and time.time() < ending_time:
            logging.info('\n[MCTS] Iteration %s', i + 1)
            logging.info('[MCTS] current parent node is %s', node.name)
            node = self.select()
            logging.info('[MCTS] selected node %s to be expanded', node.name)
            expanded_node = self.expand(parent=node)
            logging.info('[MCTS] expanded node %s to %s', node.name, expanded_node.name)
            n_wins, n_ties = self.simulate(node=expanded_node, n_simulations=n_simulations)
            self.backpropagate(node=expanded_node, n_plays=n_simulations, n_wins=n_wins, n_ties=n_ties)
            if display_tree:
                self.show_tree()
            i += 1
            logging.info('Resulting tree: %s \n', self.show_tree(return_string=True))
        logging.info('[MCTS] Performed %s iterations in %s seconds.', i, round(time.time() - starting_time, 2))

    def recommended_play(self, scoring_func=average_wins):
        """
        Move recommended by the Monte Carlo Tree Search

        :return: tuple corresponding to the recommended move
        """
        nodes = list(LevelOrderGroupIter(self.root))
        if nodes:
            level1_nodes = nodes[1]
            scores = [scoring_func(plays=n.n_plays, wins=n.n_wins, ties=n.n_ties) for n in level1_nodes]
            best_node = level1_nodes[np.argmin(scores)]  # argmin because it is the score of the other player (level 1)
            return best_node.game.last_play


def main():
    """
    Run a Monte Carlo Tree search
    """
    logging.basicConfig(level=logging.INFO)
    game = Game()
    tree = MonteCarloTreeSearch(game)
    tree.search(max_iterations=15, max_runtime=10, n_simulations=1, display_tree=False)


if __name__ == "__main__":
    main()
