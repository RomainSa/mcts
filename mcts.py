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

    def __init__(self, board_init_params, node_init_params, **kwargs):
        self.__dict__.update(kwargs)
        self.board_init_params = board_init_params
        self.node_init_params = node_init_params
        self.root = Node('0', board=Board(**board_init_params), **node_init_params)

    def select(self, parent, method='ucb1'):
        """
        Select a node of the tree based on UCB1 confidence bounds scores or expand current one if not all children
        have been visited

        :return: Node with best score
        """
        # if parent still has unexplored children we expand one of them by selecting the parent node
        if len(parent.board.legal_plays()) > len(parent.children):
            return parent
        # else we return children with the best UCB1 score
        else:
            # filter out nodes that can not be expanded
            nodes = [node for node in PreOrderIter(self.root)
                     if len(node.board.legal_plays()) > 0 and node.name != self.root.name
                     and len(node.board.legal_plays()) > len(node.children)]
            if method == 'ucb1':
                # total number of plays
                total_plays = 0
                for node in nodes:
                    total_plays += node.n_plays
                # UCB1 scores
                scores = []
                for node in nodes:
                    score = node.n_wins / max(node.n_plays, 1) + np.sqrt(
                        2 * np.log(max(total_plays, 1)) / max(node.n_plays, 1))
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
            logging.debug('Chosen node: {}'.format(nodes[np.argmax(scores)].name))
            return best

    def expand(self, parent):
        """
        Randomly choose a child for selected node in order to expand the tree

        :return: Child node
        """
        # filter out plays that already have been expanded
        already_played_moves = [node.board.last_play for node in parent.children]
        unexplored_plays = [play for play in parent.board.legal_plays() if play not in already_played_moves]
        # choose one play randomly
        selected_play = unexplored_plays[np.random.choice(len(unexplored_plays), 1)[0]]
        # create a new node where this play is performed
        child_board = deepcopy(parent.board)
        child_board.play(selected_play)
        child_name = parent.name + '_' + str(len(parent.children))
        child = Node(name=child_name, parent=parent, board=child_board, **self.node_init_params)
        logging.debug('EXPANDING play {} from node {} to new child {}'.format(selected_play, parent.name, child.name))
        return child

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
            while len(board.legal_plays()) > 0:
                board.play()
            if board.winner() == player:
                n_wins += 1
        logging.debug('SIMULATED {} plays and got {} wins'.format(n_simulations, n_wins))
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
            logging.debug('BACKPROPAGATED {} plays to node {}'.format(n_plays, ancestor.board.last_play))
            # depending on the player the number of wins is not the same
            if node.board.current_player_value == ancestor.board.current_player_value:
                ancestor.n_wins += n_wins
                logging.debug('BACKPROPAGATED {} wins to node {}'.format(n_wins, ancestor.board.last_play))
            else:
                ancestor.n_wins += (n_plays-n_wins)
                logging.debug('BACKPROPAGATED {} wins to node {}'.format((n_plays-n_wins), ancestor.board.last_play))

    def run(self, max_iterations, max_runtime, n_simulations=1):
        """
        Run a Monte Carlo Tree Search starting from root node

        :param max_iterations: max number of iterations for the tree search
        :param max_runtime: max search time in seconds
        :param n_simulations: number of simulations per expanded node
        """
        i, starting_time, ending_time = 0, time.time(), time.time() + max_runtime
        node = self.root
        while i < max_iterations and time.time() < ending_time:
            logging.debug('MCTS Iteration {}'.format(i+1))
            node = self.select(parent=node)
            expanded_node = self.expand(parent=node)
            n_wins = self.simulate(node=expanded_node, n_simulations=n_simulations)
            self.backpropagate(node=expanded_node, n_plays=n_simulations, n_wins=n_wins)
            i += 1
        logging.info('Performed {} iterations in {} seconds.'.format(i, time.time()-starting_time))

    def display(self, n_levels):
        """
        Print the tree and each state probability of play

        :return:
        """
        for i, level_nodes in enumerate(list(LevelOrderGroupIter(self.root))[1:(n_levels+1)]):
            print('ITERATION {}'.format(i))
            scores = [node.n_wins / max(node.n_plays, 1) for node in level_nodes]
            best_move = level_nodes[np.argmax(scores)]
            print('best score: {}'.format(best_move.n_wins / max(best_move.n_plays, 1)))
            print(best_move.board.display())

    def display_all(self):
        """
        Print the whole tree with probability of play

        :return:
        """
        for i, level_nodes in enumerate(LevelOrderGroupIter(self.root)):
            print('ITERATION {}'.format(i))
            for node in level_nodes:
                print("NODE {} has {} WINS and {} PLAYS.".format(node.name, node.n_wins, node.n_plays))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tree = MonteCarloTreeSearch(board_init_params={'size': 3, 'save_history': False},
                                node_init_params={'n_plays': 0, 'n_wins': 0})
    tree.run(max_iterations=1000, max_runtime=10, n_simulations=1)
    print([(n.n_wins, n.n_plays) for n in list(LevelOrderGroupIter(tree.root))[1]])
    #tree.display(n_levels=1)
    # TODO: dans le expand, on peut jouer le dernier move?
