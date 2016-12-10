def poker(hands):
    """
    Return a list of winning hands.
    """
    return allmax(hands, key=hand_rank)

def allmax(iterable, key=None):
    """
    Return a list of all items equal to
    the max of the iterable.
    """
    key = key or (lambda x: x)
    max_instance = max(iterable, key=key)
    return [i for i in iterable if key(i) == key(max_instance)]

def hand_rank(hand):
    """
    Return a tuple value indicate the
    value of a hand.
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
    Return a shorted list of card ranks
    in a hand in reverse order.
    """
    ranks = ['--23456789TJQKA'.index(r) for r,_ in hand]
    ranks.sort(reverse=True)
    return [5,4,3,2,1] if (ranks == [14, 5, 4, 3, 2]) else ranks

def flush(hand):
    """
    Return True if all the cards in 
    hand have the same suit.
    """
    suits = [s for r,s in hand]
    return len(set(suits)) == 1

def straight(ranks):
    """
    Return if the ranks are straight.
    """
    return (max(ranks) - min(ranks) == 4) and (len(set(ranks)) == 5)

def kind(n, ranks):
    """
    Return the first rank that occurs
    exactly n times in ranks.
    """
    for r in ranks:
        if ranks.count(r) == n:
            return r
    return None

def two_pair(ranks):
    """
    If there are two pairs, return the ordered
    tuple of that two ranks in reverse order.
    """
    pair = kind(2, ranks)
    lowpair = kind(2, list(reversed(ranks)))
    if pair and lowpair != pair:
        return (pair, lowpair)
    else:
        return None

def test():
    """
    Quick test cases.
    """
    "Test cases for the functions in poker program."
    sf1 = "6C 7C 8C 9C TC".split() # Straight Flush
    sf2 = "6D 7D 8D 9D TD".split() # Straight Flush
    fk = "9D 9H 9S 9C 7D".split() # Four of a Kind
    fh = "TD TC TH 7C 7D".split() # Full House
    assert poker([sf1, sf2, fk, fh]) == [sf1, sf2] 
    return 'tests pass'

if __name__ == '__main__':
    print(test())
