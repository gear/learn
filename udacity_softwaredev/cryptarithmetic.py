# python=3.5.2

import itertools, re, time, cProfile

def solve(formula):
    """
    Given a formula 'ODD + ODD == EVEN', fill in digits
    to solve it.
    """
    for f in fill_in(formula):
        if valid(f):
            return f
    
def fill_in(formula):
    """
    Generate all possible filling-in of letters in
    formula with digits.
    """
    letters = "".join(set(re.findall('[A-Z]', formula)))
    for digits in itertools.permutations('1234567890', len(letters)):
        table = str.maketrans(letters, ''.join(digits))
        yield formula.translate(table)

def valid(f):
    """
    Formula f is valid if it has no numbers
    with leading zero, and evals true.
    """
    try:
        return not re.search(r'\b0[0-9]', f) and eval(f) is True
    except ArithmeticError:
        return False

examples = """TWO + TWO == FOUR
CAT + DOG == WAR
A**2 + B**2 == C**2
X / X == X
A**N + B**N == C**N and N > 1
ATOM**0.5 == A + TO + M
sum(range(AA)) == BB
PLUTO not in set([PLANETS])""".splitlines()

def test(): 
    for puzzle in examples:
        print("Question:", puzzle)
        print("Answer:", compile_word(puzzle))
        print('')

def compile_word(word):
    """
    Compile a word of uppercase letters as numeric digits.
    E.g., compile_word('YOU') => '(1*U+10*O+100*Y)'
    Non-uppercase words unchanged: compile_word('+') => '+'
    """
    compiled = []
    var = []
    for c in word:
        if c.isupper():
            var.append(c)
        else:
            if len(var) > 0:
                compiled.append('(' + '+'.join('1' + i*'0' + '*' + ch for i, ch in enumerate(var[::-1])) + ')')
                var.clear()
            compiled.append(c)
    if len(var) > 0:
        compiled.append('(' + '+'.join('1' + i*'0' + '*' + ch for i, ch in enumerate(var[::-1])) + ')')
    return ''.join(compiled)

if __name__ == '__main__':
    test()
