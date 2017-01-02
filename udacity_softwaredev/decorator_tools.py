"""coding=utf-8
python=3.5.2
func=decorators""" 


from functools import update_wrapper


def decorator(d):
    """Make a function d a decorator."""
    def _d(fn):
        return update_wrapper(d(fn), fn)
    update_wrapper(_d, d)
    return _d


@decorator
def countcalls(f):
    """Decorator that makes the function count the function
    calls to it."""
    def _f(fn):
        countcalls[fn]


@decorator
def trace(f):
    indent = '  '
    def _f(*args):
        signature = '%s(%s)' % (f.__name__, ', '.join(map(repr,args)))
        print('%s--> %s' % (trace.level*indent, signature))
        trace.level += 1
        try:
            result = f(*args) 
            print('%s<-- %s === %s' % ((trace.level-1)*indent,
                                      signature, result))
        finally:
            trace.level -= 1
            return result
    trace.level = 0
    return _f


@trace
def fib(n):
    if n == 0 or n == 1:
        return 1
    else:
        return fib(n-1) + fib(n-2)


def test():
    print('Fibonanci 6:')
    fib(6)


if __name__ == '__main__':
    test()
