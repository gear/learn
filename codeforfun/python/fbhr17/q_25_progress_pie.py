from math import hypot, degrees, acos

wsize  = (100,100)
radius = min(wsize)/2
center = (wsize[0]/2, wsize[1]/2)
v0 = (0, radius)
epsilon = 1e-6

def dist(x, y):
    return hypot(x-center[0], y-center[1])

def angle(v1, v2):
    return degrees(acos(dot(v1, v2) / (hypot(*v1) * hypot(*v2))))

def dot(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]

def which_color(x, y, P):
    v1 = (x-center[0], y-center[1])
    d = dist(x, y)
    a = angle(v1, v0) 
    if d > radius:
        return "white"
    else:
        if x < center[0]:
            a += 180 
        if a < (P / 100.0) * 360.0:
            return "black"
        else:
            return "white"

def test(input_file="progress_pie.txt"):
    result = []
    with open(input_file, 'r') as f:
        num_case = int(f.readline())
        for case in range(num_case):
            P, x, y = map(float, f.readline().strip().split())
            result.append("Case #{}: {}".format(case+1, which_color(x, y, P)))
    with open('progess_pie_out.txt', 'w') as f:
        for r in result:
            f.write(r+'\n')

if __name__ == "__main__":
    test()
