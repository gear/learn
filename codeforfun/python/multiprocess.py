import numpy as np

from multiprocessing import Pool
from time import time
import threading

def f(x):
    return x*x

if __name__ == '__main__':
    t1 = threading.Thread(target=f)
    t1.start()
    t1.join()
