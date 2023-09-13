import numba
import numpy as np


@numba.experimental.jitclass([
    ('type', numba.intp),
    ('min_point', numba.optional(numba.float64[:])),
    ('max_point', numba.optional(numba.float64[:])),
    ('centroid', numba.float64[:])
])
class AABB:
    def __init__(self, min_point=None, max_point=None):
        if min_point is None:
            min_point = np.array([np.inf, np.inf, np.inf])
        if max_point is None:
            max_point = np.array([-np.inf, -np.inf, -np.inf])

        self.min_point = np.minimum(min_point, max_point)
        self.max_point = np.maximum(min_point, max_point)

        self.centroid = (min_point+max_point)/2