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
    if plays == 0:
        return 99.   # to be sure that arm is played at least once
    else:
        return wins / plays


def ucb1(plays, wins, ties, total_plays, c_=1.0):
    """
    Upper Confidence Bound score

    :param plays: number of times the arm has been played
    :param ties: number of ties
    :param wins: number of successes
    :param total_plays: number of plays of all arms
    :param c_: constant (the more the larger the bound)
    :return: score (min:0, max:1)
    """
    if plays == 0:
        return 99.   # to be sure that arm is played at least once
    else:
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
    if plays == 0:
        return 99.   # to be sure that arm is played at least once
    else:
        return beta.rvs(a=wins+1, b=plays-wins+1, size=1)[0]
