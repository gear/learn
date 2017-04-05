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

def square(x): return x*x
sqrt = inverse(square)
