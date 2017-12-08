"""
Monte Carlo Tree Search implementation
"""

import logging
import time
from copy import deepcopy

import numpy as np
from anytree import Node, LevelOrderGroupIter, RenderTree
from scipy.stats import beta

from games.tictactoe import Board


class MonteCarloTreeSearch:
    """
    Implementation of the Monte Carlo Tree Search algorithm
    Based on http://mcts.ai/pubs/mcts-survey-master.pdf
    """

    def __init__(self, node_init_params, board_init_params=None, board_initialisation=None, **kwargs):
        self.__dict__.update(kwargs)
        self.node_init_params = node_init_params
        if board_initialisation is not None:
            self.board_init_params = {'size': board_initialisation.size,
                                      'save_history': board_initialisation.save_history}
            self.root = Node('0', board=board_initialisation, **node_init_params)
        else:
            self.board_init_params = board_init_params
            self.root = Node('0', board=Board(**board_init_params), **node_init_params)

    def select(self, method='ucb1'):
        """
        Select a node of the tree based on UCB1 confidence bounds scores or expand current one if not all children
        have been visited

        :return: Node with best score
        """
        def ucb(node):
            # UCB1 score
            score = node.n_wins / max(node.n_plays, 1)\
                    + 0.5 * np.sqrt(np.log(max(node.parent.n_plays, 1)) / max(node.n_plays, 1))
            score = min(score, 1.0)
            score += np.random.rand() * 1e-6  # small random perturbation to avoid ties
            return score

        def thompson(node):
            # Thompson sampling score
            return beta.rvs(a=node.n_wins+1, b=node.n_plays-node.n_wins+1, size=1)[0]

        # selection start from root node
        node = self.root

        # we then go down the tree
        while node.children:
            if len(node.board.legal_plays()) > len(node.children):
                # if node still has unexplored children we select it
                logging.debug('-SELECT- chose node: %s that was not completely expanded', node.name)
                return node
            else:
                # we go down the tree until we reach the bottom always choosing the best score at each level
                nodes = node.children
                if method == 'ucb1':
                    for node in nodes:
                        node.score = ucb(node)
                elif method == 'thompson':
                    for node in nodes:
                        node.score = thompson(node)
                else:
                    raise ValueError('Unknown method')
                scores = [n.score for n in nodes]
                node = nodes[np.argmax(scores)]   # node with highest score
                logging.debug('-SELECT- chose temporary best score node: %s', node.name)
        logging.debug('-SELECT- final choice is node: %s', node.name)
        return node

    def expand(self, parent):
        """
        Randomly choose a child for selected node in order to expand the tree

        :return: Child node
        """
        # filter out plays that already have been expanded
        already_played_moves = [node.board.last_play for node in parent.children]
        unexplored_plays = [play for play in parent.board.legal_plays() if play not in already_played_moves]
        if unexplored_plays:
            # choose one play randomly
            selected_play = unexplored_plays[np.random.choice(len(unexplored_plays), 1)[0]]
            # create a new node where this play is performed
            child_board = deepcopy(parent.board)
            child_board.play(selected_play)
            child_name = parent.name + '_' + str(len(parent.children))
            child = Node(name=child_name, parent=parent, board=child_board, **self.node_init_params)
            logging.debug('-EXPAND- played %s from node %s to new child %s', selected_play, parent.name, child.name)
            node = child
        else:
            # if all nodes have been explored return parent without expanding (can happen at end of tree search)
            logging.debug('-EXPAND- did not expanded parrent %s', parent.name)
            node = parent
        return node

    def simulate(self, node, n_simulations):
        """
        Simulate games from current board state and returns number of wins

        :param node: node from which the simulated games start
        :param n_simulations: number of games simulations to perform
        :return: number of time the current player has won
        """
        n_wins = 0
        for _ in range(n_simulations):
            board = deepcopy(node.board)
            player = board.current_player.value
            logging.debug('-SIMULATE- from state\n%s\n with player %s', board.display(return_string=True), player)
            while board.legal_plays():
                board.play()
            if board.winner() == player:
                n_wins += 1
            logging.debug('-SIMULATE- and ended on state\n%s', board.display(return_string=True))
        logging.debug('-SIMULATE- performed %s plays and got %s wins', n_simulations, n_wins)
        return n_wins

    def backpropagate(self, node, n_plays, n_wins):
        """
        Back-propagate the results of the simulations to the ancestor nodes of the tree

        :param node:
        :param n_plays:
        :param n_wins:
        """
        # apply updates on current node and its ancestors
        node.n_plays += n_plays
        node.n_wins += n_wins
        for ancestor in node.ancestors:
            ancestor.n_plays += n_plays
            logging.debug('-BACKPROPAGATED- %s plays to ancestor node %s', n_plays, ancestor.name)
            # depending on the player the number of wins is not the same
            if node.board.current_player.value == ancestor.board.current_player.value:
                ancestor.n_wins += n_wins
                logging.debug('-BACKPROPAGATED- %s wins to same player ancestor node %s', n_wins, ancestor.name)
            else:
                ancestor.n_wins += (n_plays - n_wins)
                logging.debug('-BACKPROPAGATED- %s wins to different player ancestor node %s', n_plays - n_wins,
                              ancestor.name)

    def display(self, return_string=False):
        """
        Print the current state of the tree along with some statistics on nodes

        :return: tree representation as a string (useful for debugging) or nothing if printed
        """
        def sort_by_move(nodes):
            return sorted(nodes, key=lambda n: n.board.last_play)

        result = ['\n']
        for p, _, n_ in RenderTree(self.root, childiter=sort_by_move):
            result.append(('%s%s | Player %s turn | %s wins in %s plays | score: %.3f' % (p,
                                                                                          n_.board.last_play,
                                                                                          n_.board.current_player.display,
                                                                                          n_.n_wins, n_.n_plays,
                                                                                          n_.score)))
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
            n_wins = self.simulate(node=expanded_node, n_simulations=n_simulations)
            self.backpropagate(node=expanded_node, n_plays=n_simulations, n_wins=n_wins)
            if display_tree:
                self.display()
            i += 1
            logging.info('Resulting tree: %s \n', self.display(return_string=True))
        logging.info('[MCTS] Performed %s iterations in %s seconds.', i, round(time.time() - starting_time, 2))

    def recommended_play(self):
        """
        Move recommended by the Monte Carlo Tree Search

        :return: tuple corresponding to the recommended move
        """
        nodes = list(LevelOrderGroupIter(self.root))
        if nodes:
            level1_nodes = nodes[1]
            scores = [node.n_wins / node.n_plays for node in level1_nodes]
            best_node = level1_nodes[np.argmin(scores)]  # argmin because it is the score of the other player
            return best_node.board.last_play


def main():
    """
    Run a Monte Carlo Tree search
    """
    logging.basicConfig(level=logging.INFO)
    tree = MonteCarloTreeSearch(board_init_params={'size': 3, 'save_history': False},
                                node_init_params={'n_plays': 0, 'n_wins': 0, 'score': 0.})
    tree.search(max_iterations=150, max_runtime=10, n_simulations=1, display_tree=False)
    print([(n.n_wins, n.n_plays) for n in list(LevelOrderGroupIter(tree.root))[1]])
    print([n.n_wins / n.n_plays for n in list(LevelOrderGroupIter(tree.root))[1]])
    print([n.board.last_play for n in list(LevelOrderGroupIter(tree.root))[1]])
    # TODO: change current player to last player (invert all signs and then argmax for recommended_play)
    # TODO: implement a show() method that prints the tree


if __name__ == "__main__":
    main()
