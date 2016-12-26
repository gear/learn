def max_rect(histogram):
    """
    Return the tuple (start idx, end idx, area)
    of the maximum rectangle in a histogram.
    """
    curr_max = 0
    max_i = max_j = 0
    for i in range(len(histogram)):
        for j in range(i, len(histogram)):
            if area(i,j, histogram) > curr_max:
                curr_max = area(i,j, histogram) 
                (max_i, max_j) = (i, j)
    return (max_i, max_j, curr_max)

def area(i, j, histogram):
    height = min(histogram[i:j+1])
    return height * (j - i + 1)

def test():
    examples = [4, 5, 3, 6, 4, 2, 0, 4, 6, 4, 2, 0]
    print(max_rect(examples))

if __name__ == '__main__':
    test()
