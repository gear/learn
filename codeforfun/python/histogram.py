import time

def max_rect(histogram):
    """
    Return the tuple (start idx, end idx, area)
    of the maximum rectangle in a histogram.
    This solution is O(n^2).
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

def max_rect_faster(histogram):
    """
    Return the tuple (start idx, end idx, area)
    of the maximum rectangle in a histogram.
    This solution is O(nlogn) by spliting
    the histogram by removing the smallest value.
    """
    g_max = 0
    i_max = j_max = 0
    i = 0; j = len(histogram)-1
    subproblems = [(i,j)]
    for x, y in subproblems:
        area_xy = area(x,y,histogram)
        if g_max < area_xy:
            i_max = x; j_max = y; g_max = area_xy
        min_xy = min(histogram[x:y+1])
        min_xy_idx = x + histogram[x:y+1].index(min_xy)
        if x < min_xy_idx:
            subproblems.append((x, min_xy_idx-1))
        if y > min_xy_idx:
            subproblems.append((min_xy_idx+1, y))
    return (i_max, j_max, g_max)

def test():
    examples = [4, 5, 3, 6, 4, 2, 0, 4, 6, 4, 2, 10, 0, 13, 16, 3, 3, 5, 3, 12, 7, 9, 39, 19]
    t0 = time.time()
    print(max_rect(examples))
    print('Time to run max_rect:', time.time() - t0) 
    t0 = time.time()
    print(max_rect_faster(examples))
    print('Time to run max_rect_faster:', time.time() - t0)

if __name__ == '__main__':
    test()
