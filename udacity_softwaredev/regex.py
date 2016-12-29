def search(pattern, text):
    """Return true if the `pattern` appears
    anywhere in the `text`."""
    if pattern.startswith('^'):
        return match(pattern[1:], text)
    else:
        return match('.*' + pattern, text)

def match(pattern, text):
    """Return true if the `pattern` appears
    in the beginning of the `text`."""
    if pattern == '':
        return True
    elif pattern == '$':
        return (text == '')
    elif len(pattern) > 1 and pattern[1] in '*?':
        p, op, pat = pattern[0], pattern[1], pattern[2:]
        if op == '*':
            return match_star(p, pat, text)
        elif op == '?':
            if match1(p, text) and match(pat, text[1:]):
                return True
            else:
                return match(pat, text)
    else:
        return (match1(pattern[0], text) and
                match(pattern[1:], text[1:]))

def match1(p, text):
    """Return true if first character of text matches
    pattern character p."""
    if not text: return False
    return p == '.' or p == text[0]

def match_star(p, pattern, text):
    """Return true if any number of char p,
    followed pattern, matches text."""
    return (match(pattern, text) or
            (match1(p, text) and 
            match_star(p, pattern, text[1:])))

def text():
    patterns = '$;a?;.*;a*;b*;this;mistereff;a*b*c;a$;abia$;like'.split(';')
    text = ';a;ab;aaaaa;bbbbb;thiss;mister;aaac;baba;yfuyfabia;like'.split(';')
    print(len(patterns))
    print(len(text))
    for p,t in zip(patterns,text):
        print(p, t, search(p,t))

if __name__ == '__main__':
    text()
