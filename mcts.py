from copy import deepcopy
import time
import numpy as np
from scipy.stats import beta
from anytree import Node, PreOrderIter, LevelOrderGroupIter
from tictactoe import TicTacToe as Board
import logging


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

    def select(self, parent, method='ucb1'):
        """
        Select a node of the tree based on UCB1 confidence bounds scores or expand current one if not all children
        have been visited

        :return: Node with best score
        """
        # if parent still has unexplored children we expand one of them by selecting the parent node
        if len(parent.board.legal_plays()) > len(parent.children):
            logging.debug('-SELECT- chose parent node: {}'.format(parent.name))
            return parent
        # else we return node with the highest UCB1 score
        else:
            # filter out terminal nodes, root node and nodes that can not be expanded
            nodes = [node for node in PreOrderIter(self.root)
                     if len(node.board.legal_plays()) > 0 and node.name != self.root.name
                     and len(node.board.legal_plays()) > len(node.children)]
            if len(nodes) > 0:
                if method == 'ucb1':
                    # UCB1 scores
                    scores = []
                    for node in nodes:
                        score = node.n_wins / max(node.n_plays, 1) +\
                                3 * np.sqrt(2 * np.log(max(node.parent.n_plays, 1)) / max(node.n_plays, 1))
                        score += np.random.rand() * 1e-6  # small random perturbation to avoid ties
                        scores.append(score)
                elif method == 'thompson':
                    successes = [node.n_wins+1 for node in nodes]
                    fails = [node.n_plays-node.n_wins+1 for node in nodes]
                    scores = beta.rvs(a=successes, b=fails, size=len(nodes)).tolist()
                else:
                    raise ValueError('Unknown method')
                best = nodes[np.argmax(scores)]
                # return node with highest score
                logging.debug('-SELECT- chose best score node: {}'.format(nodes[np.argmax(scores)].name))
                return best
            else:
                # if all nodes have been explored randomly select one (can happen at end of tree search)
                random_node = np.random.choice(list(PreOrderIter(self.root)), 1)[0]
                logging.debug('-SELECT- chose random node: {}'.format(random_node.name))
                return random_node

    def expand(self, parent):
        """
        Randomly choose a child for selected node in order to expand the tree

        :return: Child node
        """
        # filter out plays that already have been expanded
        already_played_moves = [node.board.last_play for node in parent.children]
        unexplored_plays = [play for play in parent.board.legal_plays() if play not in already_played_moves]
        if len(unexplored_plays) > 0:
            # choose one play randomly
            selected_play = unexplored_plays[np.random.choice(len(unexplored_plays), 1)[0]]
            # create a new node where this play is performed
            child_board = deepcopy(parent.board)
            child_board.play(selected_play)
            child_name = parent.name + '_' + str(len(parent.children))
            child = Node(name=child_name, parent=parent, board=child_board, **self.node_init_params)
            logging.debug('-EXPAND- played {} from node {} to new child {}'.format(selected_play, parent.name, child.name))
            return child
        else:
            # if all nodes have been explored return parent without expanding (can happen at end of tree search)
            logging.debug('-EXPAND- did not expanded parrent {}'.format(parent.name))
            return parent

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
            player = board.current_player_value
            logging.debug('-SIMULATE- from state\n{}\n with player {}'.format(board.display(return_string=True),
                                                                              player))
            while len(board.legal_plays()) > 0:
                board.play()
            if board.winner() == player:
                n_wins += 1
            logging.debug('-SIMULATE- and ended on state\n{}'.format(board.display(return_string=True)))
        logging.debug('-SIMULATE- performed {} plays and got {} wins'.format(n_simulations, n_wins))
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
            logging.debug('-BACKPROPAGATED- {} plays to ancestor node {}'.format(n_plays, ancestor.name))
            # depending on the player the number of wins is not the same
            if node.board.current_player_value == ancestor.board.current_player_value:
                ancestor.n_wins += n_wins
                logging.debug('-BACKPROPAGATED- {} wins to same player ancestor node {}'.format(n_wins, ancestor.name))
            else:
                ancestor.n_wins += (n_plays-n_wins)
                logging.debug('-BACKPROPAGATED- {} wins to different player ancestor node {}'.format((n_plays-n_wins), ancestor.name))

    def search(self, max_iterations, max_runtime, n_simulations=1):
        """
        Run a Monte Carlo Tree Search starting from root node

        :param max_iterations: max number of iterations for the tree search
        :param max_runtime: max search time in seconds
        :param n_simulations: number of simulations per expanded node
        """
        i, starting_time, ending_time = 0, time.time(), time.time() + max_runtime
        node = self.root
        while i < max_iterations and time.time() < ending_time:
            logging.info('\n[MCTS] Iteration {}'.format(i+1))
            logging.info('[MCTS] current parent node is {}'.format(node.name))
            node = self.select(parent=node)
            logging.info('[MCTS] selected node {} to be expanded'.format(node.name))
            expanded_node = self.expand(parent=node)
            logging.info('[MCTS] expanded node {} to {}'.format(node.name, expanded_node.name))
            n_wins = self.simulate(node=expanded_node, n_simulations=n_simulations)
            self.backpropagate(node=expanded_node, n_plays=n_simulations, n_wins=n_wins)
            i += 1
        logging.info('[MCTS] Performed {} iterations in {} seconds.'.format(i, round(time.time()-starting_time, 2)))

    def recommended_play(self):
        """
        Move recommended by the Monte Carlo Tree Search

        :return: tuple corresponding to the move to play
        """
        nodes = list(LevelOrderGroupIter(self.root))
        if len(nodes) > 0:
            level1_nodes = nodes[1]
            scores = [node.n_wins / node.n_plays for node in level1_nodes]
            best_node = level1_nodes[np.argmin(scores)]   # argmin because it is the score of the other player
            return best_node.board.last_play


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    tree = MonteCarloTreeSearch(board_init_params={'size': 3, 'save_history': False},
                                node_init_params={'n_plays': 0, 'n_wins': 0})
    tree.search(max_iterations=15, max_runtime=10, n_simulations=1)
    print([(n.n_wins, n.n_plays) for n in list(LevelOrderGroupIter(tree.root))[1]])
    print([n.n_wins / n.n_plays for n in list(LevelOrderGroupIter(tree.root))[1]])
    print([n.board.last_play for n in list(LevelOrderGroupIter(tree.root))[1]])
    # TODO: change current player to last player (invert all signs and then argmax for recommended_play)
    # TODO: implement a show() method that prints the tree
