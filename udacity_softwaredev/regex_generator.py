#coding=utf-8
#python=3.5.2
#author=gear
#date=20161231

"""Set of pattern generator which takes a pattern
and generate string with length specified in the
set Ns."""

def lit(s):         return lambda Ns: set([s]) if len(s) in Ns else null
def alt(x,y):       return lambda Ns: x(Ns) | y(Ns)
def star(s):        return lambda Ns: opt(plus(x))(Ns)
def plus(x):        return lambda Ns: genseq(x, star(x), Ns, start=1)
def oneof(chars):   return lambda Ns: set().union(i for c in chars for i in lit(c)(Ns))
def seq(x, y):      return lambda Ns: genseq(x, y, Ns)
def opt(x):         return alt(epsilon, x)
dot = oneof('?')
epsilon = lit('')

null = frozenset()

def test():

    f = lit('hello')
    assert f(set([1,2,3,4,5])) == set(['hello'])
    assert f(set([1,2,3,4])) == null
    
    g = alt(lit('hi'), lit('bye'))
    assert g(set([1,2,3,4,5,6])) == set(['bye', 'hi'])
    assert g(set([1,3,5])) == set(['bye'])
    
    h = oneof('theseletters')
    assert h(set([1,2,3])) == set(['t', 'h', 'e', 's', 'l', 'r'])
    assert h(set([2,3,4])) == null

    return 'tests pass'    

if __name__ == '__main__':
    print(test())
