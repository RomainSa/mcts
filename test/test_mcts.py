import unittest
import logging
from copy import deepcopy

from anytree import Node, PreOrderIter, LevelOrderGroupIter
from anytree.search import findall

from mcts import MonteCarloTreeSearch
from games.tictactoe import Game


class TestMCTSMethods(unittest.TestCase):

    def make_level(self, starting_node, n_nodes):
        possible_plays = starting_node.game.legal_plays()
        for i in range(n_nodes):
            new_game = deepcopy(starting_node.game)
            move = possible_plays.pop()
            new_game.play(move)
            new_name = starting_node.name + '_' + str(i)
            _ = Node(name=new_name, parent=starting_node, game=new_game, **self.tree.node_init_params)

    def update_nodes(self, n_plays, n_wins, n_ties=0, nodes=None, node_names=None, include_ancestors=False):
        # get selection as nodes
        selected_nodes = []
        if node_names is not None and nodes is None:
            for node_name in node_names:
                for node in findall(self.tree.root, filter_=lambda n: node_name == n.name):
                    selected_nodes.append(node)
        elif node_names is None and nodes is not None:
            selected_nodes = nodes
        else:
            raise ValueError('One of node_names and nodes arguments must be None')

        # get ancestors if needed
        nodes_to_update = []
        if include_ancestors:
            for node in selected_nodes:
                for ancestor in node.ancestors:
                    nodes_to_update.append(ancestor)
                nodes_to_update.append(node)
        else:
            for node in selected_nodes:
                nodes_to_update.append(node)

        # update corresponding nodes
        for node in nodes_to_update:
            node.n_plays = n_plays
            node.n_wins = n_wins
            node.n_ties = n_ties

    def setUp(self):
        # tree init
        self.game = Game()
        self.tree = MonteCarloTreeSearch(game=self.game)
        # 1st level (complete)
        self.make_level(self.tree.root, 9)
        # 2nd level (complete)
        self.make_level(self.tree.root.children[3], 8)
        # 3rd level (incomplete)
        self.make_level(self.tree.root.children[3].children[4], 3)

    def test_select_level1(self):
        # create fake wins and plays
        self.update_nodes(nodes=list(PreOrderIter(self.tree.root)), n_plays=1000, n_wins=2)

        # give best attributes to one node
        node_name = '0_6'
        self.update_nodes(node_names=[node_name], n_plays=50, n_wins=45)

        selected_node = self.tree.select()   # to update scores
        self.assertEqual(node_name, selected_node.name)

    def test_select_level2(self):
        # create fake wins and plays
        self.update_nodes(nodes=list(PreOrderIter(self.tree.root)), n_plays=1000, n_wins=2)

        # give best attributes to ancestors of one node
        node_name = '0_3_6'
        self.update_nodes(node_names=[node_name], include_ancestors=True, n_plays=50, n_wins=45)

        selected_node = self.tree.select()   # to update scores
        self.assertEqual(node_name, selected_node.name)

    def test_select_level3(self):
        # create fake wins and plays
        self.update_nodes(nodes=list(PreOrderIter(self.tree.root)), n_plays=1000, n_wins=2)

        # give best attributes to ancestors of one node
        node_name = '0_3_4_1'
        node = findall(self.tree.root, filter_=lambda n: n.name == node_name)[0]
        self.update_nodes(node_names=[node_name], include_ancestors=True, n_plays=50, n_wins=45)

        selected_node = self.tree.select()   # to update scores
        self.assertEqual(node.parent.name, selected_node.name)   # because node is not fully expanded

    def test_expand(self):
        # expand one node
        node = self.tree.root.children[3].children[4]
        n_nodes_before_expand = len(list(PreOrderIter(self.tree.root)))
        self.tree.expand(node)
        n_nodes_after_expand = len(list(PreOrderIter(self.tree.root)))

        self.assertEqual(n_nodes_before_expand+1, n_nodes_after_expand)

    def test_simulate(self):
        n_wins, _ = self.tree.simulate(node=self.tree.root, n_simulations=100)
        self.assertGreater(n_wins, 1)

    def test_backpropagate1(self):
        # select a node from which backpropagation start
        node_name = '0_3_4_1'
        node = findall(self.tree.root, filter_=lambda n: n.name == node_name)[0]
        total_plays_before = sum([n.n_plays for n in PreOrderIter(self.tree.root)])

        # perform backpropagation
        self.tree.backpropagate(node, n_plays=100, n_wins=10, n_ties=0)
        total_plays_after = sum([n.n_plays for n in PreOrderIter(self.tree.root)])
        total_wins_after = sum([n.n_wins for n in PreOrderIter(self.tree.root)])

        self.assertEquals(total_plays_after-total_plays_before, 4*100)
        self.assertGreater(total_wins_after, 1)

    def test_backpropagate2(self):
        # create fake wins and plays
        self.update_nodes(nodes=list(PreOrderIter(self.tree.root)), n_plays=0, n_wins=0)

        # select a node from which backpropagation start
        node_name = '0_3_6'
        node = findall(self.tree.root, filter_=lambda n: n.name == node_name)[0]
        self.tree.backpropagate(node, n_plays=20, n_wins=6, n_ties=0)

        # root node stats
        root_plays = self.tree.root.n_plays
        root_wins = self.tree.root.n_wins

        # level 1 nodes stats
        level1_nodes = list(LevelOrderGroupIter(self.tree.root))[1]
        level1_plays = sum([node.n_plays for node in level1_nodes])
        level1_wins = sum([node.n_wins for node in level1_nodes])

        self.assertEquals(root_plays, level1_plays)
        self.assertEquals(root_plays-root_wins, level1_wins)

    def test_recommend(self):
        # create fake wins and plays
        self.update_nodes(nodes=list(PreOrderIter(self.tree.root)), n_plays=100, n_wins=1)

        # backpropagate good attributes to node where same player than root has to play
        node_name = '0_3_4'
        node = findall(self.tree.root, filter_=lambda n: node_name == n.name)[0]
        self.tree.backpropagate(node, n_plays=1000, n_wins=999, n_ties=0)

        # corresponding level1 name
        node_to_recommend_name = '_'.join(node_name.split('_')[:2])
        node_to_recommend = findall(self.tree.root, filter_=lambda n: node_to_recommend_name == n.name)[0]

        # move corresponding to node
        recommended_move = self.tree.recommended_play()
        self.assertEqual(node_to_recommend.game.last_play, recommended_move)


if __name__ == '__main__':
    unittest.main()
