import numba
import numpy as np

from primitives.ray import Ray
from utils.constants import ONES, ZEROS, BLUE
from utils.misc import hit_object
from utils.vectors import normalize

from numba_progress import ProgressBar


@numba.njit(nogil=True, parallel=True)
def render_scene(scene, primitives, bvh, progress_proxy):
    for y in numba.prange(scene.height):
        color = ZEROS
        progress_proxy[0].update()
        for x in numba.prange(scene.width):

            offset_x = ((x + 0.5) / scene.width - 0.5) * 2.0
            offset_y = ((y + 0.5) / scene.height - 0.5) * 2.0

            ray_direction = scene.look_at + np.array([offset_x, offset_y, 0], dtype=np.float64)
            ray_origin = scene.camera

            ray = Ray(ray_origin, ray_direction, 0)

            nearest_object, min_distance, intersection, surface_normal = hit_object(primitives, bvh, ray)

            if nearest_object is not None:
                scene.image[y, x] = [0, 0, 1]

            progress_proxy[1].update(1)

        progress_proxy[1].set(0)

    return scene.image


@numba.njit(nogil=True, parallel=True)
def _render_scene(scene, primitives, bvh):
    for y in numba.prange(scene.height):
        color = ZEROS
        for x in numba.prange(scene.width):

            offset_x = ((x + 0.5) / scene.width - 0.5) * 2.0
            offset_y = ((y + 0.5) / scene.height - 0.5) * 2.0

            ray_direction = scene.look_at + np.array([offset_x, offset_y, 0], dtype=np.float64)
            ray_origin = scene.camera

            ray = Ray(ray_origin, ray_direction, 0)

            nearest_object, min_distance, intersection, surface_normal = hit_object(primitives, bvh, ray)

            if nearest_object is not None:
                scene.image[y, x] = [0, 0, 1]

        print((y/scene.height)*100)

    return scene.image