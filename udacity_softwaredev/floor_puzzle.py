import itertools

def floor_puzzle():
    for (Hopper, Kay, Liskov, Perlis, Ritchie) in itertools.permutations([1,2,3,4,5]):
        if Hopper == 5: continue
        if Kay == 1: continue
        if Liskov == 1 or Liskov == 5: continue
        if Perlis < Kay: continue
        if abs(Ritchie - Liskov) == 1: continue
        if abs(Liskov - Kay) == 1: continue
        return [Hopper, Kay, Liskov, Perlis, Ritchie]
    return None

def test():
    assert(floor_puzzle() == [3, 2, 4, 5, 1])
    print(floor_puzzle())
    print('Test passed.')

if __name__ == '__main__':
    test()
