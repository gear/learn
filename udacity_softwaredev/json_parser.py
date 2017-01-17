"""coding=utf-8
python=3.5.2
func=json_parser"""

from functools import update_wrapper
from string import split
import re


def split(string, token=None, max_split=-1):
    """Split the string, strip the whitespaces, and remove empty splits."""
    return [i.strip for i in string.strip().split(token, max_split) if i]

def grammar(description, whitespace=r'\s*'):
    """Convert a description to a grammar. Each line is a rule for a
    non-terminal symbol. For example:
        Symbol => A1 A2 ... | B1 B2 ... | C1 C2 ...
    where the rhs is one or more alternatives, separated by the `|` sign.
    Each alternative is a sequence of atoms, separated by spaces. 
    An atom is either a symbol on some lhs, or itis a regular expr
    that will be passed to re.match to match a token.

    Notation for *, +, or ? is not allowed in a rule alternative (but
    OK within a token). Use `\` to continue long lines. The space or
    tab wrapping `|` or `=>` must be provided in the grammar by default.
    To disable this, use '' as the `whitespace` argument for `grammar`."""
    G = {' ': whitespace}
    description = description.replace('\t', ' ')  # no tabs
    for line in split(description, '\n'):
        lhs, rhs = split(line, ' => ', 1) # Split into list of two at most
        alternatives = split(rhs, ' | ')
        G[lhs] = tuple(map(split, alternatives))
    return G

def decorator(d):
    """Make function d a decorator: d wraps a fucntion fn."""
    def _d(fn):
        return update_wrapper(d(fn), fn)
    update_wrapper(_d, d)
    return _d

def memo(f):
    """Decorator that caches the return value for each call
    to f(args). Then when called again with same args, we 
    can just look it up."""
    cache = {}
    def _f(*args):
        try:
            return cache[args]
        except KeyError:
            cache[args] = result = f(*args)
            return result
        except TypeError:  # Some element in args cannot be hashed
            return f(args)
    return _f

def parse(start_symbol, text, grammar):
    """Return a (tree, remainder) pair. If remainder is an
    empty string, it parsed the whole string. Failure iff 
    the remainder is None. This is a deterministic PEG parser,
    so rule order (left-to-right) matters. Do `E => T op E | T`,
    putting the longest parse first; don't do `E => T | T op E`.
    Also, no left recursion is allowed: no `E => E op T`."""
    # Tokenizer is the white space symbol and a tuple
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

JSON = grammar("""
object => [{] members [}] | [{] [}]
members => pair [,] members | pair
pair => string [:] value
array => [[] Elements []] | [[] []]
elements => Value [,] Elements | Value
value => String | Number | Object | Array | True | False | Null
string => ["] Chars ["] | ["] ["]
chars => Char Chars | Char
char => \w
number => 
""", whitespace='\s')

Fail = (None, None)

def test():
    assert json_parse('["testing", 1, 2, 3]') == (                      
                       ['value', ['array', '[', ['elements', ['value', 
                       ['string', '"testing"']], ',', ['elements', ['value', ['number', 
                       ['int', '1']]], ',', ['elements', ['value', ['number', 
                       ['int', '2']]], ',', ['elements', ['value', ['number', 
                       ['int', '3']]]]]]], ']']], '')
    
    assert json_parse('-123.456e+789') == (
                       ['value', ['number', ['int', '-123'], ['frac', '.456'], ['exp', 'e+789']]], '')
    
    assert json_parse('{"age": 21, "state":"CO","occupation":"rides the rodeo"}') == (
                      ['value', ['object', '{', ['members', ['pair', ['string', '"age"'], 
                       ':', ['value', ['number', ['int', '21']]]], ',', ['members', 
                      ['pair', ['string', '"state"'], ':', ['value', ['string', '"CO"']]], 
                      ',', ['members', ['pair', ['string', '"occupation"'], ':', 
                      ['value', ['string', '"rides the rodeo"']]]]]], '}']], '')
    print('tests pass')
    return 'test pass'

if __name__ == '__main__':
    test()
