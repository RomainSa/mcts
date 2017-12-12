import numpy as np
from scipy.stats import beta


def average_wins(plays, wins, ties):
    """
    Simple average

    :param plays: number of times the arm has been played
    :param wins: number of successes
    :param ties: number of ties
    :return: score (min:0, max:1)
    """
    plays = max(plays, 1)   # to avoid division by 0
    score = wins / plays
    return score


def ucb1(plays, wins, ties, total_plays, c_=0.5):
    """
    Upper Confidence Bound score

    :param plays: number of times the arm has been played
    :param ties: number of ties
    :param wins: number of successes
    :param c_: constant (the more the larger the bound)
    :return: score (min:0, max:1)
    """
    plays = max(plays, 1)   # to avoid division by 0
    total_plays = max(total_plays, 1)

    score = wins / plays + c_ * np.sqrt(np.log(total_plays) / plays)
    score = min(score, 1.0)
    score += np.random.rand() * 1e-6  # small random perturbation to avoid ties
    return score


def thompson(plays, wins, ties):
    """
    Thompson sampling

    :param plays: number of times the arm has been played
    :param wins: number of successes
    :param ties: number of ties
    :return: score (min:0, max:1)
    """
    score = beta.rvs(a=wins+1, b=plays-wins+1, size=1)[0]
    return score
