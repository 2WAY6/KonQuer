from math import sqrt

# import numpy as np
# from numba import jit


def dist_2d(A, B):
    return sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)


def dist_3d(A, B):
    return sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2 + (A[2] - B[2]) ** 2)


# >= because of zero area case
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) >= (B[1] - A[1]) * (C[0] - A[0])


# @jit
# # CCW (Counter Clock Wise check) ONLY WORKS IN 2D OF COURSE
# # TODO: Source?
# def ccw_jit(A, B, C):
#     return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


# # Segment Intersection -> Intersection includes shared point
# @jit
# def segments_intersect_jit(S1, S2):
#     s1_a, s1_b, s2_a, s2_b = S1[0], S1[1], S2[0], S2[1]
#     return (ccw_jit(s1_a, s2_a, s2_b) != ccw_jit(s1_b, s2_a, s2_b)
#             and ccw_jit(s1_a, s1_b, s2_a) != ccw_jit(s1_a, s1_b, s2_b))


# Math
def det(A, B):  # dt.: Determinante
    return A[0] * B[1] - A[1] * B[0]


# Line intersection (ATTENTION: infinitely long segment !!!).
# Also includes shared point
def line_intersection(S1, S2,):
    S1A, S1B, S2A, S2B = S1[0], S1[1], S2[0], S2[1]

    x_diff = [S1A[0] - S1B[0], S2A[0] - S2B[0]]
    y_diff = [S1A[1] - S1B[1], S2A[1] - S2B[1]]

    div = det(x_diff, y_diff)

    assert div != 0
    # if div == 0:
    #     return None

    d = [det(S1A, S1B), det(S2A, S2B)]
    x = det(d, x_diff) / div
    y = det(d, y_diff) / div

    return (x, y)


# check if point N is in closed polygon
def point_in_element(N, poly):
    bools = []
    cnt = len(poly)
    for i in range(cnt):
        i1 = i
        i2 = (i+1) % cnt
        bools.append(ccw(N, poly[i1], poly[i2]))
    if sum(bools) == cnt or sum(bools) == 0:
        return True
    else:
        return False
