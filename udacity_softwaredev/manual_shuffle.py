# utf-8
# Python 3.5.2
# Test simple shuffling algorithms

import time
import random

def bad_shuffle(lst):
    is_swapped = []
    for _ in lst:
        is_swapped.append(False)
    while not all(i == True for i in is_swapped):
        i, j = random.randint(0, len(lst)-1), random.randint(0, len(lst)-1)
        swap(lst, i, j)
        is_swapped[i] = is_swapped[j] = True
    return tuple(lst)

def peter_shuffle(lst):
    """
    Donald Knuth called this P algorithm.
    """
    for i, _ in enumerate(lst):
        j = random.randint(i, len(lst)-1)
        swap(lst, i, j)
    return tuple(lst)

def swapall_shuffle(lst):
    """
    Swap with all other indices
    """
    if len(lst) <= 1:
        return tuple(lst)
    j = 0
    for i, _ in enumerate(lst):
        j = random.randint(0, len(lst)-1)
        swap(lst, i, j)
    return tuple(lst)
    
def swap(lst, i, j):
    (lst[i], lst[j]) = (lst[j], lst[i])

def test(shuff_func, lst, num_shuffle=5000):
    abc_count = {}
    for _ in range(num_shuffle):
        abc = lst.copy() # note this
        shuff = shuff_func(abc)
        if shuff in abc_count:
            abc_count[shuff] += 1
        else:
            abc_count[shuff] = 1
    print('Shuffle result:')
    print('\n'.join(str(i) for i in abc_count.items()))

if __name__ == '__main__':
    abc_list = ['A', 'B', 'C']
    bigger_list = [i for i in range(50)]
    print('Teacher shuffle:')
    test(bad_shuffle, abc_list)
    print('Peter shuffle:')
    test(peter_shuffle, abc_list)
    print('Swap all shuffle:')
    test(swapall_shuffle, abc_list)
