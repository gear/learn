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
    sorted_ranks = sorted(['--23456789TJQKA'.index(r) for r,s in hand], \
                          reverse = True)
    unzip = lambda lst: zip(*lst)
    counts, ranks = unzip(group(shorted_ranks))
    if ranks == (14, 5, 4, 3, 2):
        ranks = (5, 4, 3, 2, 1)
    straight = len(ranks) == 5 and max(ranks) - min(ranks) == 4
    flush = len(set([s for _, s in hand])) == 1
    return (9 if (5,) == counts else
            8 if straight and flush else
            7 if (4,1) == counts else
            6 if (3,2) == counts else
            5 if flush else
            4 if straight else
            3 if (3, 1, 1) == counts else
            2 if (2, 2, 1) == counts else
            1 if (2, 1, 1, 1) == counts else
            0), ranks 

def group(items):
    """
    Return item count zipped with the unique item.
    """
    return [(items.count(i), i) for i in set(items)] 


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
