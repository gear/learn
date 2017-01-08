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
     

json_grammar = r"""
Object => [{] Members [}] | [{] [}]
Members => Pair [,] Members | Pair
Pair => String [:] Value
Array => [[] Elements []] | [[] []]
Elements => Value [,] Elements | Value
Value => String | Number | Object | Array | True | False | Null
String => ["] Chars ["] | ["] ["]
Chars => Char Chars | Char
Char => \w
Number => -?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?
"""
