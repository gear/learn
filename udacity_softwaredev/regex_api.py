from functools import update_wrapper

def search(pattern, text):
    """Match pattern anywhere in text;
    return the longest earliest match or None."""
    for i in range(len(text)):
        m = match(pattern, text[i:])
        if m is not None:
            return m

def match(pattern, text):
    """Match pattern against start of text;
    return longest match found or None."""
    remainders = matchset(pattern, text)
    if remainders:
        shortest = min(remainders, key=len)
        return text[:len(text)-len(shortest)] if shortest is not None else None
    else:
        return None

def matchset(pattern, text):
    """Match pattern at start of text;
    return a set of remainders of text."""
    op, x, y = components(pattern)
    if 'lit' == op:
        return set([text[len(x):]]) if text.startswith(x) else null
    elif 'seq' == op:
        return set(t2 for t1 in matchset(x, text) for t2 in matchset(y, t1))
    elif 'alt' == op:
        return matchset(x, text) | matchset(y, text)
    elif 'dot' == op:
        return set([text[1:]]) if text else null
    elif 'oneof' == op:
        return set([text[1:]]) if text.startswith(x) else null # Check startswith API
    elif 'eol' == op:
        return set(['']) if text == '' else null
    elif 'star' == op:
        return (set([text]) | 
                set(t2 for t1 in matchset(x, text) 
                    for t2 in matchset(pattern, t1) if t1 != text))
    else:
        raise ValueError('Unknown pattern: %s' % pattern)

null = frozenset() 

def components(pattern):
    """Decompose a pattern into operator, x and y."""
    x = pattern[1] if len(pattern) > 1 else None
    y = pattern[2] if len(pattern) > 2 else None
    return pattern[0], x, y

def decorator(d):
    def _d(fn):
        return update_wrapper(d(fn), fn)
    update_wrapper(_d, d)
    return _d

@decorator
def n_ary(f):
    """Given a binary function f(x,y), return an n_ary
    function such that f(x,y,z) = f(x, f(y,z)), etc."""
    def n_ary_f(x, *args):
        return x if not args else f(x, n_ary_f(*args))
    return n_ary_f

def lit(string):    return ('lit', string)
@n_ary
def seq(x, y):      return ('seq', x, y)
def alt(x, y):      return ('alt', x, y)
def star(x):        return ('star', x)
def plus(x):        return seq(x, star(x))
def opt(x):         return alt(lit(''), x)
def oneof(chars):   return ('oneof', tuple(chars))
dot = ('dot',)
eol = ('eol',)

def test(t_matchset=True, t_api=True, t_matchsearch=True):
    """Quick test for matchset function."""
    if t_matchset:
        assert matchset(('lit', 'abc'), 'abcdef') == set(['def'])
        assert matchset(('dot',), 'am i missing something?') == set(['m i missing something?'])
        assert matchset(('oneof', 'a'), 'aabc123') == set(['abc123'])
        print('matchset test passes')
    if t_api:
        assert lit('abc') == ('lit', 'abc')
        assert seq(('lit', 'a'),
                   ('lit', 'b')) == ('seq', ('lit', 'a'), ('lit', 'b'))
        assert star(('lit', 'a')) == ('star', ('lit', 'a'))
        assert oneof('abc') == ('oneof', ('a','b','c'))
        print('api test passes')
    if t_matchsearch:
        assert match(('star', ('lit', 'a')), 'aaabcd') == 'aaa'
        assert match(('alt', ('lit', 'b'), ('lit', 'c')), 'ab') == None
        assert match(('alt', ('lit', 'b'), ('lit', 'a')), 'ab') == 'a'
        assert search(('alt', ('lit', 'b'), ('lit', 'c')), 'ab') == 'b'
        print('match and search test passes')

if __name__ == '__main__':
    test()
