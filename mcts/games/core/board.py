"""
Board class
"""


import numpy as np


class Board:

    def __init__(self, shape=None, state=None, dtype=np.int8):
        """
        Board constructor.
        Shape or state must be provided. If both are then only state will be used.

        :param shape: board shape
        :type shape: tuple
        :param state: initial board state
        :type state: np.array
        """
        if shape is not None and state is None:
            self.state = np.zeros(shape, dtype=dtype)
        elif state is not None:
            self.state = state
        else:
            raise ValueError('Shape or initial state must be provided')

    def __str__(self):
        """ Board representation as a string """
        # state is a numpy array
        return self.state.__str__()
