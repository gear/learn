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
PLUTO not in set([PLANETS])
A + B == BA""".splitlines()

samples = """YOU
ME
DRAGON
KNIGHT""".splitlines()

def test(): 
    for puzzle in examples:
        print("Question:", puzzle)
        print("Answer:", fast_solve(puzzle))
        print('')

def compile_word(word):
    if word.isupper():
        formula = '+'.join('1%s*%s' % (i*'0', c) for i, c in enumerate(word[::-1]))
        return '(' + formula + ')'
    else: 
        return word

def compile_formula(formula, verbose=False):
    letters = ''.join(set(re.findall('[A-Z]', formula)))
    params = ', '.join(letters)
    tokens = map(compile_word, re.split(r'([A-Z]+)', formula))
    leading_letters = [word[0] for word in re.findall(r'[A-Z]+', formula) if len(word) > 1]
    leading_checks = ''.join(['False if %s == 0 else ' % (i) for i in leading_letters])
    body = ''.join(tokens)
    f = 'lambda %s: %s' % (params, leading_checks + body)
    if verbose: print(f)
    return eval(f), letters

def fast_solve(formula):
    f, letters = compile_formula(formula)
    for digits in itertools.permutations((1,2,3,4,5,6,7,8,9,0), len(letters)):
        try:
            if f(*digits) is True:
                table = str.maketrans(letters, ''.join(map(str, digits)))
                return formula.translate(table)
        except ArithmeticError:
            pass

if __name__ == '__main__':
    test()
