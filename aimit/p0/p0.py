import numpy as np
from sys import argv

def main():
  func = globals()[argv[1]]
  try:
    la = argv[2]
    print(func(argv[2]))
  except:
    print(func())
 
def cube(n):
  n = float(n)
  return n*n*n

def factorial(n):
  n = float(n)
  assert n > -1, "factorial: input must not be negative"
  if n == 0:
    return 1
  return n * factorial(n-1)

def count_pattern(patt = ('a', 'k', 'c'), lst = ('a', 'b', 'c', 'd')):
  if not isinstance(lst, (list, tuple)):
    raise ValueError("Input list should be list or tuple.")
  if not isinstance(patt, (list, tuple)):
    raise ValueError("Input pattern should be list or tuple.")
  count = 0
  for i in range(len(lst) - len(patt) + 1):
    for j in range(len(patt)):
      if not lst[i+j] == patt[j]:
        count -= 1
        break
    count += 1
  return count
 
def depth(expr_list = (('/', ('expt', 'x', 5), 
                      ('expt', ('-', ('expt', 'x', 2),1), ('/', 5, 2)))) ):
  if not isinstance(expr_list, (list, tuple)):
    return 0
  else:
    return 1 + max([depth(x) for x in expr_list]) 

def tree_ref(tree=(((1, 2), 3), (4, (5, 6)), 7, (8, 9, 10)), indices=(3,1)):
  pointer = tree
  for i in indices:
    pointer = pointer[i]
  return pointer

if __name__ == '__main__':
  main()
