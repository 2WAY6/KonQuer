# cython: profile=True
from cython cimport boundscheck, wraparound
import numpy as np
from  libc.math cimport fmin, fmax


# Returns determinant of two 2d vectors
@boundscheck(False)
@wraparound(False)
cdef double det_2d(double[:] A, double[:] B):
    return A[0] * B[1] - A[1] * B[0]


# Check, if three 2d points are in a counter clockwise order
@boundscheck(False)
@wraparound(False)
cdef bint is_ccw(double[:] A, double[:] B, double[:] C):
    cdef double[2] CA
    CA[0] = A[0] - C[0]
    CA[1] = A[1] - C[1]

    cdef double[2] CB = [B[0] - C[0], B[1] - C[1]]
    # cdef double[:] CB = np.array([B[0] - C[0], B[1] - C[1]])

    cdef double det = det_2d(CA, CB)
    #return 1 if det > 0 else 0 # Corner case det == 0 is not checked for yet
    return True if det > 0 else False # Corner case det == 0 is not checked for yet


# check for intersection of segment AB and segment CD
@boundscheck(False)
@wraparound(False)
cpdef bint segments_intersect(double[:] A, double[:] B, double[:] C, double[:] D):
    if max(A[0], B[0]) < min(C[0], D[0]):
        return False
    elif min(A[0], B[0]) > max(C[0], D[0]):
        return False
    elif max(A[1], B[1]) < min(C[1], D[1]):
        return False
    elif min(A[1], B[1]) > max(C[1], D[1]):
        return False

    cdef bint o1 = is_ccw(A, B, C)
    cdef bint o2 = is_ccw(A, B, D)
    cdef bint o3 = is_ccw(C, D, A)
    cdef bint o4 = is_ccw(C, D, B)

    if o1 != o2 and o3 != o4:
        return True

    else:  # ignoring colinear cases for the moment
        return False


# polygon has to be closed. meaning first point == last point
@boundscheck(False)
@wraparound(False)
cdef bint is_point_in_polygon(double[:] P, double[:] R, double[:,:] polygon, double ray_length=10000000):
    cdef int n_vertices = polygon.shape[0]
    # cdef double[:] R = np.array([P[0] + ray_length, P[1]])
    # cdef double[2] R = (P[0] + ray_length, P[1])

    cdef int cnt = 0
    cdef int i
    for i in range(n_vertices - 1):
        if segments_intersect(P, R, polygon[i], polygon[i+1]):
            cnt += 1

    if cnt%2 == 0:
        return False
    else:
        return True


@boundscheck(False)
@wraparound(False)
cpdef int assign_points_to_polygons(double[:,:] points, list polygons, long[:] points_conn_polygons, double[:,:] polygon_bboxes):
    cdef int n_polys = len(polygons)
    cdef int i
    cdef int n_points = points.shape[0]

    for i in range(n_points):
        points_conn_polygons[i] = assign_point_to_polygon(points[i], polygons, n_polys, polygon_bboxes)

    return 1


@boundscheck(False)
@wraparound(False)
cdef int assign_point_to_polygon(double[:] P, list polygons, int n_polys, double[:,:] polygon_bboxes, double ray_length=10000000):
    cdef int i
    cdef double[:] R = np.array([P[0] + ray_length, P[1]])

    for i in range(n_polys):
        if polygon_bboxes[i, 0] < P[0] < polygon_bboxes[i, 1] and polygon_bboxes[i, 2] < P[1] < polygon_bboxes[i, 3]:
            if is_point_in_polygon(P, R, polygons[i]):
                return i

    return -1