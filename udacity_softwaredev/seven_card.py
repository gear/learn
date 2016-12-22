# coding: utf-8
# 2016-12-19
# CS 212, hw1-1, 7-card stud

import itertools

def best_hand(hand):
    """
    From a 7-card hand, return the best
    5-card hand.
    """
    return max(itertools.combinations(hand, 5), 
               key=hand_rank)

"""
Test code
"""

def test_best_hand():
    assert (sorted(best_hand("6C 7C 8C 9C TC 5C JS".split()))
            == ['6C', '7C', '8C', '9C', 'TC'])
    assert (sorted(best_hand("TD TC TH 7C 7D 8C 8S".split()))
            == ['8C', '8S', 'TC', 'TD', 'TH'])
    assert (sorted(best_hand("JD TC TH 7C 7D 7S 7H".split()))
            == ['7C', '7D', '7H', '7S', 'JD'])
    return 'test_best_hand passes'

def main():
    print(test_best_hand())

"""
Provided functions
"""

def hand_rank(hand):
    """
    Return a value indicating the ranking of a hand.
    """
    ranks = card_ranks(hand)
    if straight(ranks) and flush(hand):
        return (8, max(ranks))
    elif kind(4, ranks):
        return (7, kind(4, ranks), kind(1, ranks))
    elif kind(3, ranks) and kind(2, ranks):
        return (6, kind(3, ranks), kind(2, ranks))
    elif flush(hand):
        return (5, ranks)
    elif straight(ranks):
        return (4, max(ranks))
    elif kind(3, ranks):
        return (3, kind(3, ranks), ranks)
    elif two_pair(ranks):
        return (2, two_pair(ranks), ranks)
    elif kind(2, ranks):
        return (1, kind(2, ranks), ranks)
    else:
        return (0, ranks)

def card_ranks(hand):
    """
    Return a list of the ranks, shorted with higher
    rank first.
    """
    ranks = ['--23456789TJQKA'.index(r) for r, _ in hand]
    ranks.sort(reverse=True)
    return [5, 4, 3, 2, 1] if (ranks == [14, 5, 4, 3, 2]) else ranks

def straight(ranks):
    """
    Return true of the ranks are consecutive.
    """
    return (max(ranks) - min(ranks) == 4) and len(set(ranks)) == 5

def flush(hand):
    """
    Return true if all ranks are the same.
    """
    suits = [s for _, s in hand]
    return len(set(suits)) == 1

def kind(n, ranks):
    """
    Return the rank that has exactly n count in hand.
    """
    for r in ranks:
        if ranks.count(r) == n:
            return r
    return None

def two_pair(ranks):
    """
    Return the sorted tuple of two pairs.
    """
    rev_ranks = sorted(ranks)
    high_pair = kind(2, ranks)
    low_pair = kind(2, rev_ranks)
    if high_pair and low_pair != high_pair:
        return (high_pair, low_pair)
    else:
        return None

"""
Run
"""
if __name__ == '__main__':
    main()
