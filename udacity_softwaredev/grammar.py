"""coding=utf-8
python=3.5.2
func=grammar"""

from functools import update_wrapper
import re


def parse(start_symbol, text, grammar):
    """Parse a text follows a grammar.
    start_symbol is the starting text of the language
    text is the remainding text
    grammar is a dictionary containing syntax

    Example:
    >>> parse('Exp', '3*x + b', G)
    This function should return a (tree, remainder) pair.
    If the remainder is '', it parsed the whole string.
    Failure iff remainder is None. This is a deterministic PEG
    parser so rule order (left-to-right) matters. Do 'E => T op E | T',
    putting the longest parse first; don't do 'E => T | T op E'
    Also, no left recursion allowed: don't do 'E => E op T'"""

    tokenizer = grammar[' '] + '(%s)'

    def parse_sequence(sequence, text):
        result = []
        for atom in sequence:
            tree, text = parse_atom(atom, text)
            if text is None: return Fail
            result.append(tree)
        return result, text

    @memo
    def parse_atom(atom, text):
        if atom in grammar:
            for alternative in grammar[atom]:
                tree, rem = parse_sequence(alternative, text)
                if rem is not None: return [atom]+tree, rem
            return Fail
        else:
            m = re.match(tokenizer % atom, text)
            return Fail if (not m) else (m.group(1), text[m.end():])

    return parse_atom(start_symbol, text)

Fail = (None, None)

def decorator(d):
    """Make function d a decorator: d wraps a function fn."""
    def _d(fn):
        return update_wrapper(d(fn), fn)
    update_wrapper(_d, d)
    return _d

@decorator
def memo(f):
    """Decorator that caches the return value for each call 
    to f(args). Then when called again with same args, we can
    just look it up."""
    cache = {}
    def _f(*args):
        try:
            return cache[args]
        except KeyError:
            cache[args] = result = f(*args)
            return result
        except TypeError:  # args can't be a key
            return f[args]
    return _f

raw_G = r"""
Exp     => Term [+-] Exp | Term
Term    => Factor [*/] Term | Factor
Factor  => Funcall | Var | Num | [(] Exp [)]
Funcall => Var [(] Exps [)]
Exps    => Exp [,] Exps | Exp
Var     => [a-zA-Z_]\w*
Num     => [-+]?[0-9]+([.][0-9]*)?
"""

def grammar(G, whitespace='\s*'):
    g_dict = {' ': whitespace}
    for rules in G.strip().split('\n'):
        lhs, rhs = rules.split(' => ')
        g_dict[lhs.strip()] = tuple(a.split() for a in rhs.split(' | '))
    return g_dict 

def test():
    print(grammar(raw_G))

if __name__ == '__main__':
    test()
