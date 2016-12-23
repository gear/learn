import itertools
import time

def print_bracket_permutations(n):
    """
    print all possible bracket
    permutation with n bracket
    """
    assert n % 2 == 0, "n must be even"
    brac = '()'
    indices = [i % 2 for i in range(n)]
    for perm in set(itertools.permutations(indices)):
        brac_config = [brac[i] for i in perm]
        if check_brac(brac_config):
            print(''.join(brac_config))

def check_brac(brackets):
    brac = '()'
    sum = 0
    for i in brackets:
        sum += 1 if i == '(' else (-1)
        if sum < 0:
            return False
    return True

def test():
    print_bracket_permutations(4)
    print_bracket_permutations(6)
    t0 = t1 = time.time()
    print_bracket_permutations(10)
    t1 = time.time()
    print('Time for 10 brackets: ',t1 - t0)

if __name__ == '__main__':
    test()
