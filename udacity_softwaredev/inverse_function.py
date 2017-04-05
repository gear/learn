# Inverse a function using Newton's method.
# The function starts from 0 and monotonically increasing


def inverse(f, delta=1/128):
    def _f(y):
        x = 0.0
        while True:
            k = (f(x + delta) - f(x)) / delta
            x = (y - f(x)) / k + x
            if f(x) - delta <= y and f(x) + delta >= y:
                return x if (f(x)-y < y-f(x-delta)) else x-delta
    return _f


def bis_inverse(f, delta=1/1024.):
    """Invert a function using binary search."""
    def _f(y):
        lo, hi = find_bounds(f, y)
        x = binary_search(lo, hi, f, y, delta)
        return x
    return _f


def find_bounds(f, y):
    x = 1.0
    while f(x) < y:
        x = x * 2
    return x/2, x


def binary_search(lo, hi, f, y, delta):
    x = (lo + hi) / 2.0
    if f(x) - delta <= y and f(x) + delta >= y:
        return x
    else:
        if f(x) > y:
            return binary_search(lo, x, f, y, delta)
        else:
            return binary_search(x, hi, f, y, delta) 

def square(x): return x*x
sqrt = inverse(square)
