#coding=utf-8
#python=3.5.2
#author=gear
#date=20170101

def parse_input(filename='linear_search.input'):
    with open(filename, 'r') as f:
       num_test = int(f.readline())
       data = f.readlines()
       a_len = data[::3]
       arrs = data[1::3]
       queries = data[2::3]
       arrs = [map(int, a.split()) for a in arrs]
       queries = [int(i) for i in queries]
       return (arrs, queries)

def linear_search(n, arr):
    for i, v in enumerate(arr):
        if v == n:
            return i
    return -1
 
def test():
    arrs, queries = parse_input()
    for arr, n in zip(arrs, queries):
        print(linear_search(n, arr))
    print('Test passes')

if __name__ == '__main__':
    test()
