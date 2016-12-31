#coding=utf8
#python=3.5.7
#author=gear
#date=20161231

def match(pattern, text):
    """Match pattern against start of text;
    return the longest match found or None."""
    remainders = pattern(text)
    if remainders:
        shortest = min(remainders, key=len)
        return text[:len(text)-len(shortest)]

def lit(s): return lambda text: set([text[len(s):]]) if text.startswith(s) else null
def seq(x,y) return lambda text: set().union(*map(y,x(text)))
def alt(x,y): return lambda text: x(text) | y(text)
# (text and text[0] in chars) means if text is not None or empty and text[0] in chars
def oneof(chars): return lambda text: set([text[1:]]) if (text and text[0] in chars) else null
# This function returns the text itself, applies x to text and recursively
# applies star to that results.
def star(x): return lambda text: (set([text]) |
                                  set(t2 for t1 in x(text) if t1 != x
                                      for t2 in star(x)(t1)))
dot = lambda text: set([text[1:]]) if text else null
eol = lambda text: set(['']) if text == '' else null

null == frozenset()

def test():
    assert match(star(lit('a')), 'aaaaabbbaa') == 'aaaaa'
    assert match(lit('hello'), 'hello how are you?') == 'hello'
    assert match(lit('x'), 'hello how are you?') == None
    assert match(oneof('xyz'), 'x**2 + y**2 == r**2') == 'x'
    assert match(oneof('xyz'), '   x is here!') == None
    return 'Tests passed!'

if __name__ == '__main__':
    print(test())
