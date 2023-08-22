import numba


@numba.experimental.jitclass([
    ('type', numba.intp),
    ('min_point', numba.float64[:]),
    ('max_point', numba.float64[:]),
    ('centroid', numba.float64[:])
])
class AABB:
    def __init__(self, min_point, max_point):
        self.min_point = min_point
        self.max_point = max_point
        self.centroid = (min_point+max_point)/2