import numba
import numpy as np

from utils.constants import EPSILON
from primitives.intersects import aabb_intersect, triangle_intersect, intersect_bounds
from primitives.triangle import Triangle
from primitives.aabb import AABB
from utils.stl4py import nth_element, partition


class BoundedBox:
    def __init__(self, prim, n):
        self.prim = prim
        self.prim_num = n
        self.bounds = get_bounds(prim)


node_type = numba.deferred_type()


@numba.experimental.jitclass([
    ('bounds', numba.optional(AABB.class_type.instance_type)),
    ('child_0', numba.optional(node_type)),
    ('child_1', numba.optional(node_type)),
    ('split_axis', numba.optional(numba.intp)),
    ('first_prim_offset', numba.optional(numba.intp)),
    ('n_primitives', numba.optional(numba.intp))
])
class BVHNode:
    def __init__(self):
        self.bounds = None
        self.child_0 = None
        self.child_1 = None
        self.split_axis = None
        self.first_prim_offset = None
        self.n_primitives = None

    def init_leaf(self, first, n, box):
        self.first_prim_offset = first
        self.n_primitives = n
        self.bounds = box

    def init_interior(self, axis, c0, c1):
        self.child_0 = c0
        self.child_1 = c1
        self.bounds = enclose_volumes(c0.bounds, c1.bounds)
        self.split_axis = axis
        self.n_primitives = 0


node_type.define(BVHNode.class_type.instance_type)


@numba.experimental.jitclass([
    ('bounds', numba.optional(AABB.class_type.instance_type)),
    ('primitives_offset', numba.optional(numba.intp)),
    ('second_child_offset', numba.optional(numba.intp)),
    ('n_primitives', numba.optional(numba.intp)),
    ('axis', numba.optional(numba.intp))
])
class LinearBVHNode:
    def __init__(self):
        self.bounds = None
        self.primitives_offset = None
        self.second_child_offset = None
        self.n_primitives = None
        self.axis = None
        # self.pad = [0]


class BucketInfo:
    def __init__(self):
        self.count = 0
        self.bounds = None


def get_bounds(prim):
    min_p = np.minimum.reduce([prim.vertex_1, prim.vertex_2, prim.vertex_3])
    max_p = np.maximum.reduce([prim.vertex_1, prim.vertex_2, prim.vertex_3])
    bounded_box = AABB(min_p, max_p)

    return bounded_box


@numba.njit
def enclose_volumes(box_1, box_2):
    if box_1 is not None:
        if box_2 is None:
            bounded_box = box_1
        else:
            min_p = np.minimum(box_1.min_point, box_2.min_point)
            max_p = np.maximum(box_1.max_point, box_2.max_point)
            bounded_box = AABB(min_p, max_p)
    else:
        bounded_box = box_2
    return bounded_box


def enclose_centroids(box, cent):
    if box is not None:
        min_p = np.minimum(box.min_point, cent)
        max_p = np.maximum(box.max_point, cent)
        bounded_box = AABB(min_p, max_p)
    else:
        bounded_box = AABB(cent, cent)

    return bounded_box


def get_largest_dim(box):
    dx = abs(box.max_point[0] - box.min_point[0])
    dy = abs(box.max_point[1] - box.min_point[1])
    dz = abs(box.max_point[2] - box.min_point[2])
    if dx > dy and dx > dz:
        return 0
    elif dy > dz:
        return 1
    else:
        return 2


def get_surface_area(box):
    diagonal = box.max_point - box.min_point
    surface_area = 2 * (diagonal[0] * diagonal[1] + diagonal[0] * diagonal[2] + diagonal[1] * diagonal[2])
    return surface_area


def offset_bounds(bounds, point):
    o = point - bounds.min_point
    if bounds.max_point[0] > bounds.min_point[0]:
        o[0] /= bounds.max_point[0] - bounds.min_point[0]

    if bounds.max_point[1] > bounds.min_point[1]:
        o[1] /= bounds.max_point[1] - bounds.min_point[1]

    if bounds.max_point[2] > bounds.min_point[2]:
        o[2] /= bounds.max_point[2] - bounds.min_point[2]

    return o


def partition_pred(x, n_buckets, centroid_bounds, dim, min_cost_split_bucket):
    b = n_buckets * offset_bounds(centroid_bounds, x.bounds.centroid)[dim]
    if b == n_buckets:
        b = n_buckets - 1
    return b <= min_cost_split_bucket


def build_bvh(primitives, bounded_boxes, start, end, ordered_prims, total_nodes):
    split_method = 2  # 0: surface area heuristics, 1: mid point, 2: alternative median
    n_boxes = len(bounded_boxes)
    max_prims_in_node = int(0.1 * n_boxes)
    max_prims_in_node = max_prims_in_node if max_prims_in_node < 10 else 10
    node = BVHNode()
    total_nodes += 1
    bounds = None
    for i in range(start, end):
        bounds = enclose_volumes(bounds, bounded_boxes[i].bounds)

    # print(start, end)

    if start == end:
        return node, bounded_boxes, ordered_prims, total_nodes

    n_primitives = end - start
    if n_primitives == 1:
        # create left bvh node
        first_prim_offset = len(ordered_prims)
        for i in range(start, end):
            prim_num = bounded_boxes[i].prim_num
            ordered_prims.append(primitives[prim_num])
        node.init_leaf(first_prim_offset, n_primitives, bounds)
        return node, bounded_boxes, ordered_prims, total_nodes
    elif n_primitives == 0:
        #TODO: Check: start == end
        first_prim_offset = len(ordered_prims)
        prim_num = bounded_boxes[start].prim_num
        ordered_prims.append(primitives[prim_num])
        node.init_leaf(first_prim_offset, n_primitives, bounds)
        return node, bounded_boxes, ordered_prims, total_nodes
    else:
        centroid_bounds = None
        for i in range(start, end):
            centroid_bounds = enclose_centroids(centroid_bounds, bounded_boxes[i].bounds.centroid)
        dim = get_largest_dim(centroid_bounds)
        # Partition primitives into two sets and build children
        mid = (start + end) // 2
        if centroid_bounds.max_point[dim] == centroid_bounds.min_point[dim]:
            # Create leaf BVH node
            first_prim_offset = len(ordered_prims)
            for i in range(start, end):
                prim_num = bounded_boxes[i].prim_num
                ordered_prims.append(primitives[prim_num])

            node.init_leaf(first_prim_offset, n_primitives, bounds)
            return node, bounded_boxes, ordered_prims, total_nodes

        else:
            if split_method == 0:
                # Partition primitives based on Surface Area Heuristic
                if n_primitives <= 4:
                    # Partition primitives into equally sized subsets
                    mid = (start + end) // 2
                    # nth_element(bounded_boxes, mid, first=start, last=end, key=lambda x: x.bounds.centroid[dim])
                    bounded_boxes[start:end] = sorted(bounded_boxes[start:end], key=lambda x: x.bounds.centroid[dim],
                                                      reverse=False)
                else:
                    n_buckets = 12
                    buckets = [BucketInfo() for _ in range(n_buckets)]
                    # Initialize BucketInfo for SAH partition buckets
                    for i in range(start, end):
                        b = n_buckets * offset_bounds(centroid_bounds, bounded_boxes[i].bounds.centroid)[dim]
                        b = int(b)
                        if b == n_buckets:
                            b = n_buckets - 1
                        buckets[b].count += 1
                        buckets[b].bounds = enclose_volumes(buckets[b].bounds, bounded_boxes[i].bounds)

                    # compute cost for splitting each bucket
                    costs = []
                    for i in range(n_buckets - 1):
                        b0 = b1 = None
                        count_0 = 0
                        count_1 = 0
                        for j in range(i + 1):
                            b0 = enclose_volumes(b0, buckets[j].bounds)
                            count_0 += buckets[j].count
                        for j in range(i + 1, n_buckets):
                            b1 = enclose_volumes(b1, buckets[j].bounds)
                            count_1 += buckets[j].count

                        _cost = .125 * (
                                    count_0 * get_surface_area(b0) + count_1 * get_surface_area(b1)) / get_surface_area(
                            bounds)
                        costs.append(_cost)

                    # find bucket to split at which minimizes SAH metric
                    min_cost = costs[0]
                    min_cost_split_bucket = 0
                    for i in range(1, n_buckets - 1):
                        if costs[i] < min_cost:
                            min_cost = costs[i]
                            min_cost_split_bucket = i

                    # Either create leaf or split primitives at selected SAH bucket
                    leaf_cost = n_primitives
                    if n_primitives > max_prims_in_node or min_cost < leaf_cost:
                        # pmid = partition(bounded_boxes,
                        #                  lambda x: partition_pred(x, n_buckets, centroid_bounds, dim, min_cost_split_bucket),
                        #                  first=start,
                        #                  last=end)
                        pmid = partition(bounded_boxes[start:end],
                                         lambda x: partition_pred(x, n_buckets, centroid_bounds, dim,
                                                                  min_cost_split_bucket))
                        mid = pmid + start  # bounded_boxes[0]
                    else:
                        # Create leaf BVH Node
                        first_prim_offset = len(ordered_prims)
                        for i in range(start, end):
                            prim_num = bounded_boxes[i].prim_num
                            ordered_prims.append(primitives[prim_num])
                        node.init_leaf(first_prim_offset, n_primitives, bounds)
                        return node, bounded_boxes, ordered_prims, total_nodes

            elif split_method == 1:
                # partition primitives through node's midpoint
                pmid = (centroid_bounds.min_point[dim] + centroid_bounds.max_point[dim]) / 2
                mid_ptr = partition(bounded_boxes[start:end],
                                    lambda x: x.bounds.centroid[dim] < pmid)
                mid = mid_ptr + start

                # if mid!=start and mid!=end:
                #     break

        # print(start, mid, end)

        child_0, bounded_boxes, ordered_prims, total_nodes = build_bvh(primitives, bounded_boxes, start, mid,
                                                                       ordered_prims, total_nodes)

        child_1, bounded_boxes, ordered_prims, total_nodes = build_bvh(primitives, bounded_boxes, mid, end,
                                                                       ordered_prims, total_nodes)

        node.init_interior(dim, child_0, child_1)

    return node, bounded_boxes, ordered_prims, total_nodes


@numba.njit
def flatten_bvh(linear_nodes, node, offset):
    """performs a depth-first traversal and
    stores the nodes in memory in linear order"""
    linear_nodes[offset].bounds = node.bounds
    _offset = offset
    if node.n_primitives > 0:
        # leaf node
        linear_nodes[offset].primitives_offset = node.first_prim_offset
        linear_nodes[offset].n_primitives = node.n_primitives
    else:
        # create interior flattened bvh node
        linear_nodes[offset].axis = node.split_axis
        linear_nodes[offset].n_primitives = 0
        # TODO: fix this
        linear_nodes, offset = flatten_bvh(linear_nodes, node.child_0, offset + 1)
        linear_nodes, linear_nodes[_offset].second_child_offset = flatten_bvh(linear_nodes, node.child_1, offset + 1)
        offset = linear_nodes[_offset].second_child_offset

    return linear_nodes, offset


# @numba.njit
# def intersect_bvh(ray_origin, ray_direction, linear_bvh, primitives):
#     triangle = None
#     current_t = np.inf
#     current_idx = 0
#     inv_dir = 1/ray_direction
#     dir_is_neg = [inv_dir[0] < 0, inv_dir[1] < 0, inv_dir[2] < 0]
#     visited_nodes = [False for _ in range(len(linear_bvh))]
#     while True:
#         if not visited_nodes[current_idx]:
#             visited_nodes[current_idx] = True
#             node = linear_bvh[int(current_idx)]
#             # print("checking node: "+ str(current_idx))
#             if intersect_bounds(node.bounds, ray_origin, ray_direction, inv_dir, dir_is_neg):
#                 # check if it's a leaf node
#                 if node.n_primitives>0:
#                     # print("intersected node: "+ str(current_idx))
#                     for i in range(node.primitives_offset, node.primitives_offset+node.n_primitives):
#                         t = triangle_intersect(ray, primitives[i])
#                         if t is None:
#                             continue
#                         if EPSILON < t < current_t:
#                             current_t = t
#                             triangle = primitives[i]
#                 else:
#                     if dir_is_neg[node.axis]:
#                         current_idx = node.second_child_offset
#                     else:
#                         for i in range(len(visited_nodes)):
#                             if not visited_nodes[i]:
#                                 current_idx = i
#                                 break
#             else:
#                 for i in range(len(visited_nodes)):
#                     if not visited_nodes[i]:
#                         current_idx = i
#                         break
#         else:
#             for i in range(len(visited_nodes)):
#                 if not visited_nodes[i]:
#                     current_idx = i
#                     break
#         all_visited = True
#         for i in range(len(visited_nodes)):
#             if not visited_nodes[i]:
#                 all_visited=False
#                 break
#         if all_visited:
#             break
#
#     return triangle, current_t


@numba.njit
def __intersect_bvh(ray, primitives, linear_bvh):
    triangles = []

    inv_dir = 1 / ray.direction

    # _id = np.random.randint(1000)

    dir_is_neg = [inv_dir[0] < 0, inv_dir[1] < 0, inv_dir[2] < 0]
    to_visit_offset = 0
    current_node_index = 0
    nodes_to_visit = [0 for _ in range(64)]  #
    # visited_nodes = [False for _ in range(len(linear_bvh))]

    while True:
        node = linear_bvh[int(current_node_index)]
        # print(str(_id)+"Current Node: "+str(current_node_index)+"\n")
        # print(str(_id)+"To Visit Offset: "+str(to_visit_offset)+"\n")
        if intersect_bounds(node.bounds, ray, inv_dir):
            # if aabb_intersect(ray_origin, ray_direction, node.bounds):
            if node.n_primitives > 0:
                # print(str(_id)+"Primitives found at: "+str(current_node_index)+"\n")
                for i in range(node.n_primitives):
                    t = triangle_intersect(ray, primitives[node.primitives_offset + i])
                    if t is not None:
                        triangles.append(primitives[node.primitives_offset + i])
                if to_visit_offset == 0:
                    # print(str(_id)+"Break due to visit offset zero \n")
                    break
                to_visit_offset -= 1
                current_node_index = nodes_to_visit[to_visit_offset]
                # print(str(_id)+"From if, next: "+str(current_node_index)+"\n")
            else:
                if dir_is_neg[node.axis]:
                    nodes_to_visit[to_visit_offset] = current_node_index + 1
                    to_visit_offset += 1
                    current_node_index = node.second_child_offset
                    # print(str(_id)+"Direction is negative, next: "+str(current_node_index)+"\n")
                else:
                    nodes_to_visit[to_visit_offset] = node.second_child_offset
                    to_visit_offset += 1
                    current_node_index += 1
                    # print(str(_id)+"Direction is positive, next: "+str(current_node_index)+"\n")
        else:
            if to_visit_offset == 0:
                break

            to_visit_offset -= 1
            current_node_index = nodes_to_visit[to_visit_offset]
            # print(str(_id)+"From else, next: "+str(current_node_index)+"\n")

    return triangles


@numba.njit
def intersect_bvh(ray, primitives, linear_bvh):
    current_idx = 0
    triangle = None
    visited = [False for _ in range(len(linear_bvh))]
    inv_dir = 1 / ray.direction
    dir_is_neg = [inv_dir[0] < 0, inv_dir[1] < 0, inv_dir[2] < 0]
    min_distance = ray.tmax

    while True:
        if not visited[current_idx]:
            node = linear_bvh[int(current_idx)]
            visited[current_idx] = True
            if intersect_bounds(node.bounds, ray, inv_dir):
                if node.n_primitives > 0:
                    # leaf node
                    for i in range(node.n_primitives):
                        leaf_idx = node.primitives_offset + i
                        visited[leaf_idx] = True
                        t = triangle_intersect(ray, primitives[leaf_idx])
                        if t is not None and EPSILON < t < min_distance:
                            min_distance = t
                            # triangles.append(primitives[leaf_idx])
                            triangle = primitives[leaf_idx]
                    if current_idx == 0:
                        # no interior nodes
                        break
                    all_visited = True
                    for i in range(len(visited)):
                        if not visited[i]:
                            all_visited = False
                            current_idx = i
                    if all_visited:
                        # print("All visited!")
                        break
                else:
                    # interior node
                    if dir_is_neg[node.axis]:
                        current_idx = node.second_child_offset
                    else:
                        current_idx += 1
            else:
                if current_idx == 0:
                    # ray doesn't intersect the tree
                    break
                all_visited = True
                for i in range(len(visited)):
                    if not visited[i]:
                        all_visited = False
                        current_idx = i
                if all_visited:
                    # print("All visited!")
                    break
        else:
            all_visited = True
            for i in range(len(visited)):
                if not visited[i]:
                    all_visited = False
                    current_idx = i
            if all_visited:
                # print("All visited!")
                break

    # print("No of triangles:-"+str(len(triangles)))
    return triangle, min_distance
