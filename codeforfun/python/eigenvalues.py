"""
coding=utf-8
python=3.5.2
func='Practicing graph operations'
"""

import numpy as np

def trace(matrix):
    trc = 0
    for i, row in enumerate(matrix):
        for j, e in enumerate(row): 
            if i == j:
                trc += e
    return trc

def test():
    mat = np.array([[1,2,3,5], [4,6,3,9], [5,2,6,8], [7,5,8,4]], dtype=np.float32)
    print('Testing trace function...')
    print(trace([[1,2,3,4],[5,4,3,2],[7,6,4,2],[1,0,9,8]]))
    assert trace(mat) == mat.trace()
    print('Test trace passed.')
    

if __name__ == '__main__':
    test()
