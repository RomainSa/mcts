"""
Player class
"""


class Player:

    def __init__(self, name, value, display):
        self.name = name
        self.value = value
        self.display = display
        assert isinstance(value, int)

    def __eq__(self, other):
        """ Two players are equal if they have the same value"""
        if isinstance(self, other.__class__):
            return self.value == other.value
        return False

    def __str__(self):
        """ Player representation as a string """
        return self.display

    def __repr__(self):
        """ Player representation as a string """
        return 'Player(name=%s, value=%s, repr=%s)' % (self.name, self.value, self.display)
