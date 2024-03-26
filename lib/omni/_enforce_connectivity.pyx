#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
from libc.float cimport DBL_MAX

import numpy as np
cimport numpy as cnp

cnp.import_array()

def _enforce_connectivity_labels_360(Py_ssize_t[:, :, ::1] segments,
                                       Py_ssize_t min_size,
                                       Py_ssize_t max_size,
                                       Py_ssize_t start_label=0):
    """ Helper function to remove small disconnected regions from the labels

    Parameters
    ----------
    segments : 3D array of int, shape (Z, Y, X)
        The label field/superpixels found by SLIC.
    min_size : int
        The minimum size of the segment
    max_size : int
        The maximum size of the segment. This is done for performance reasons,
        to pre-allocate a sufficiently large array for the breadth first search
    start_label : int
        The label indexing start value.

    Returns
    -------
    connected_segments : 3D array of int, shape (Z, Y, X)
        A label field with connected labels starting at label=1
    """

    # get image dimensions
    cdef Py_ssize_t depth, height, width
    depth = segments.shape[0]
    height = segments.shape[1]
    width = segments.shape[2]

    # neighborhood arrays
    cdef Py_ssize_t[::1] ddx = np.array((1, -1, 0, 0, 0, 0), dtype=np.intp)
    cdef Py_ssize_t[::1] ddy = np.array((0, 0, 1, -1, 0, 0), dtype=np.intp)
    cdef Py_ssize_t[::1] ddz = np.array((0, 0, 0, 0, 1, -1), dtype=np.intp)

    # new object with connected segments initialized to mask_label
    cdef Py_ssize_t mask_label = start_label - 1

    cdef Py_ssize_t[:, :, ::1] connected_segments \
        = np.full_like(segments, mask_label, dtype=np.intp)

    cdef Py_ssize_t current_new_label = start_label
    cdef Py_ssize_t label = start_label

    # variables for the breadth first search
    cdef Py_ssize_t current_segment_size = 1
    cdef Py_ssize_t bfs_visited = 0
    cdef Py_ssize_t adjacent

    cdef Py_ssize_t zz, yy, xx

    cdef Py_ssize_t[:, ::1] coord_list = np.empty((max_size, 3), dtype=np.intp)

    with nogil:
        for z in range(depth):
            for y in range(height):
                for x in range(width):

                    if segments[z, y, x] == mask_label:
                        continue

                    if connected_segments[z, y, x] > mask_label:
                        continue

                    # find the component size
                    adjacent = current_new_label
                    label = segments[z, y, x]
                    connected_segments[z, y, x] = current_new_label
                    current_segment_size = 1
                    bfs_visited = 0
                    coord_list[bfs_visited, 0] = z
                    coord_list[bfs_visited, 1] = y
                    coord_list[bfs_visited, 2] = x

                    #perform a breadth first search to find
                    # the size of the connected component
                    while bfs_visited < current_segment_size < max_size:
                        for i in range(6):
                            zz = coord_list[bfs_visited, 0] + ddz[i]
                            yy = coord_list[bfs_visited, 1] + ddy[i]
                            xx = coord_list[bfs_visited, 2] + ddx[i]
                            if (0 <= yy < height and 0 <= zz < depth):

                                #Horizontal circular effect
                                if xx < 0:
                                    xx += width  
                                elif xx >= width:
                                    xx -= width 
    
                                if (segments[zz, yy, xx] == label and
                                    connected_segments[zz, yy, xx] == mask_label):
                                    connected_segments[zz, yy, xx] = \
                                        current_new_label
                                    coord_list[current_segment_size, 0] = zz
                                    coord_list[current_segment_size, 1] = yy
                                    coord_list[current_segment_size, 2] = xx
                                    current_segment_size += 1
                                    if current_segment_size >= max_size:
                                        break
                                elif (connected_segments[zz, yy, xx] > mask_label and
                                      connected_segments[zz, yy, xx] != current_new_label):
                                    adjacent = connected_segments[zz, yy, xx]
                        bfs_visited += 1

                    # change to an adjacent one, like in the original paper
                    if current_segment_size < min_size:
                        for i in range(current_segment_size):
                            connected_segments[coord_list[i, 0],
                                               coord_list[i, 1],
                                               coord_list[i, 2]] = adjacent
                    else:
                        current_new_label += 1

    return np.asarray(connected_segments)
