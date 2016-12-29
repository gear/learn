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
        return # dot api 
    elif 'oneof' == op:
        return # oneof api
    elif 'eol' == op:
        return set(['']) if text == '' else null
    elif 'star' == op:
        return (set([text]) | 
                set(t2 for t1 in matchset(x, text)
                    for t2 matchset(pattern, t1) if t1 != text))
    else:
        raise ValueError('Unknown pattern: %s' % pattern)

null = frozenset() 

