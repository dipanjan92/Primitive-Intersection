import numba
import numpy as np

from primitives.ray import Ray
from utils.constants import ONES, ZEROS, BLUE
from utils.misc import hit_object
from utils.vectors import normalize



@numba.njit
def trace_path(scene, primitives, bvh, ray, bounce, rand):
    throughput = np.ones((3), dtype=np.float64)
    light = np.zeros((3), dtype=np.float64)
    specular_bounce = False

    while True:
        # terminate path if max bounce is reached
        # if bounce>=scene.max_depth:
        #     break

        rand_0 = rand[0][bounce]
        rand_1 = rand[1][bounce]

        # intersect ray with scene
        nearest_object, min_distance, intersection, surface_normal = hit_object(primitives, bvh, ray)

        # terminate path if no intersection is found
        if nearest_object is None:
            # no object was hit
            break

        if np.dot(surface_normal, ray.direction) > 0:
            # ray_inside_object = True
            break

        light += BLUE
        print("Hit!")
        break




        # # check if diffuse surface
        # if nearest_object.material.is_diffuse:
        #     shadow_ray_origin = intersection + EPSILON * surface_normal
        #     # direct light contribution
        #     direct_light = cast_one_shadow_ray(scene, primitives, bvh, nearest_object, shadow_ray_origin, surface_normal)
        #
        #     # indirect light contribution
        #     indirect_ray_direction, pdf = cosine_weighted_hemisphere_sampling(surface_normal, ray.direction, [rand_0, rand_1])
        #
        #     if pdf==0:
        #         break
        #
        #     indirect_ray_origin = intersection + EPSILON * indirect_ray_direction
        #
        #     # change ray direction
        #     ray.origin = indirect_ray_origin
        #     ray.direction = indirect_ray_direction
        #
        #     cos_theta = np.dot(indirect_ray_direction, surface_normal)
        #
        #     brdf = nearest_object.material.color.diffuse * inv_pi
        #
        #     throughput *= brdf * cos_theta / pdf
        #
        #     indirect_light = throughput * trace_path(scene, primitives, bvh, ray, bounce+1, rand)
        #
        #     light += (direct_light+indirect_light)
        #
        #
        # elif nearest_object.material.is_mirror:
        #     # mirror reflection
        #     ray.origin = intersection + EPSILON * surface_normal
        #     ray.direction = get_reflected_direction(ray.direction, surface_normal)
        #
        # elif nearest_object.material.transmission>0.0:
        #     # compute reflection
        #     # use Fresnel
        #     if ray_inside_object:
        #         n1 = nearest_object.material.ior
        #         n2 = 1
        #     else:
        #         n1 = 1
        #         n2 = nearest_object.material.ior
        #
        #     R0 = ((n1 - n2)/(n1 + n2))**2
        #     theta = np.dot(ray.direction, surface_normal)
        #
        #     reflection_prob = R0 + (1 - R0) * (1 - np.cos(theta))**5 # Schlick's approximation
        #
        #     # ----
        #
        #     Nr = nearest_object.material.ior
        #     if np.dot(ray.direction, surface_normal)>0:
        #         Nr = 1/Nr
        #     Nr = 1/Nr
        #     cos_theta = -(np.dot(ray.direction, surface_normal))
        #     _sqrt = 1 - (Nr**2) * (1 - cos_theta**2)
        #
        #     if _sqrt > 0 and rand_0>reflection_prob:
        #         # refraction
        #         ray.origin = intersection + (-EPSILON * surface_normal)
        #         transmit_direction = (ray.direction * Nr) + (surface_normal * (Nr * cos_theta - np.sqrt(_sqrt)))
        #         ray.direction = normalize(transmit_direction)
        #
        #     else:
        #         # reflection
        #         ray.origin = intersection + EPSILON * surface_normal
        #         ray.direction = get_reflected_direction(ray.direction, surface_normal)
        #
        # else:
        #     # error
        #     break

        # # terminate path using russian roulette
        # if bounce>3:
        #     r_r = max(0.05, 1-throughput[1]) # russian roulette factor
        #     if rand_0<r_r:
        #         break
        #     throughput /= 1-r_r
        #
        # bounce += 1

    return light




@numba.njit(parallel=True)
def render_scene(scene, primitives, bvh):

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

        print(y)

    return scene.image