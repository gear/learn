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

def fast_print_bracket_perm(n):
    """
    print all possible bracket
    in linear time wrt. number of 
    possible bracket config.
    """ 
    assert n % 2 == 0, "n must be even"
    left = right = n // 2
    left -= 1
    print_brac('(', left, right)
    
def print_brac(bstr, left, right):
    if left > right:
        return
    if left == 0: 
        bstr += ')' * right
        print(bstr)
    elif left == right == 1: 
        bstr += '()'
        print(bstr)
    else:
        add_left = bstr + '('
        add_right = bstr + ')'
        print_brac(add_left, left - 1, right)
        print_brac(add_right, left, right - 1)

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
    print('Time for 10 brackets:',t1 - t0)
    t0 = t1 = time.time()
    fast_print_bracket_perm(24)
    t1 = time.time()
    print('Time for 24 brackets fast print:', t1-t0)

if __name__ == '__main__':
    test()
