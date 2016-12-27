def longest_subpalindrome_slice(text):
    """
    Return the longest substring which
    is a panlindrome.
    """
    max_len = 0
    max_range = (0,0)
    text = text.lower()
    # Odd subpalindrome
    for i, c in enumerate(text):
        w_size = min(i+1, len(text) - i) 
        sub_len = 0
        for j in range(w_size):
            if text[i + j] == text[i - j]:
                sub_len += 1
            else:
                break
        if (2*sub_len + 1) > max_len:
            max_range = (i - (sub_len - 1), i + sub_len)
            max_len = 2 * (sub_len-1) + 1
    # Even subpalindrome
    for i, c in enumerate(text):
        w_size = min(i+1, len(text) - i - 1)
        sub_len = 0
        for j in range(w_size):
            if text[i + j + 1] == text[i - j]:
                sub_len += 1
            else:
                break
        if 2*sub_len > max_len:
            max_range = (i - sub_len + 1, i + sub_len+1)
            max_len = 2 * sub_len
    return max_range

def test():
    L = longest_subpalindrome_slice
    assert L('racecar') == (0, 7)
    assert L('Racecar') == (0, 7)
    assert L('RacecarX') == (0, 7)
    assert L('something rac e car going') == (8, 21)
    assert L('Race carr') == (7, 9)
    assert L('xxxxx') == (0, 5)
    assert L('Mad am I ma dam.') == (0, 15)
    assert L('maam') == (0,4)
    return 'tests pass'

if __name__ == '__main__':
    print(test())
