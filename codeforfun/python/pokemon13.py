# Compute how much chance to complete the
# set of 13 pokemons when I buy n mystery bags
# Ref: https://en.wikipedia.org/wiki/Coupon_collector%27s_problem

import math
import sys
from time import time
from random import randint, seed

def hamonic(k):
    """Return the k-th harmonic number."""
    hnum = 0
    for i in range(1,k+1):
        hnum += 1/k 
    return hnum

def time_for(k, bound):
    """Return the number of items I need to buy
    to get a certain guarantee.""" 
    if bound > 1.0 or bound < 0.0:
        print("Invalid bound: {}".format(bound)) 
        return 0
    c = 1 / bound
    return k * hamonic(k) * c

def main():
    k = int(sys.argv[1])
    bound = float(sys.argv[2])
    print(time_for(k, bound)) 

def simulate(k):
    seed(time())
    unique = set()
    step = 0
    while len(unique) < k:
        step += 1
        unique.add(randint(1,k))
    return step 
    
if __name__ == "__main__":
    main() 
