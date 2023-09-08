'''
Standard Library Functions
'''
import numba


@numba.njit
def partition(arr, pred, first=0, last=None):
    # if last is None, consider the whole array
    if last is None:
        last = len(arr)

    # i: start of the false partition
    i = first
    for j in range(first, last):
        if pred(arr[j]):
            # if the predicate function returns true,
            # swap the current element with the first element of the false partition
            arr[i], arr[j] = arr[j], arr[i]
            i += 1

    # i is now pointing to the first false element
    return i


@numba.njit
def mid_point_partition(arr, midpoint_partition, pmid):
    left_ptr, right_ptr = 0, len(arr) - 1
    while True:
        while left_ptr <= right_ptr and midpoint_partition.mid_point_pred(arr[left_ptr], pmid):
            left_ptr += 1
        while right_ptr >= 0 and not midpoint_partition.mid_point_pred(arr[right_ptr], pmid):
            right_ptr -= 1
        if left_ptr <= right_ptr:
            arr[left_ptr], arr[right_ptr] = arr[right_ptr], arr[left_ptr]
        else:
            return left_ptr


@numba.njit
def nth_element(arr, n, first=0, last=None, key=lambda x: x):
    if last is None:
        last = len(arr) - 1
    if first < last:
        pi = partition(arr, key, first, last)
        if pi > n:
            nth_element(arr, n, first, pi - 1, key)
        elif pi < n:
            nth_element(arr, n, pi + 1, last, key)
        else:
            return arr[n]