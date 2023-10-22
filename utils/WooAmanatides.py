# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
This file includes the implementation of a fast and simple voxel traversal
algorithm by Woo and Amanatides. The algorithm is described in the paper
"Fast Voxel Traversal Algorithm for Ray Tracing" by Woo and Amanatides
(http://www.cse.yorku.ca/~amana/research/grid.pdf).

Author(s):  Enrico Stefanel (enrico.stefanel@studenti.units.it)
Date:       2023-10-03
"""

def WooLineAlgorithm(region, point_a, point_b):
    """ This function implements the Woo and Amanatides algorithm for
    checking if a line segment intersects a region of a matrix.
    Input(s):   region:     a matrix of booleans
                point_a:    the coordinates of the first point of the segment
                point_b:    the coordinates of the second point of the segment
    Output(s):  True if the segment intersects the region, False otherwise """

    # Get the dimensions of the region matrix
    rows, cols = region.shape

    # Extract the coordinates of points A and B
    y1, x1 = point_a
    y2, x2 = point_b

    # Ensure the coordinates are integers
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Check if the segment is out of bounds
    if (
        x1 < 0 or x1 >= rows or x2 < 0 or x2 >= rows or
        y1 < 0 or y1 >= cols or y2 < 0 or y2 >= cols
    ):
        return False  # Out of bounds, no intersection

    # Check if the segment passes through any True cells in the region
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while x1 != x2 or y1 != y2:

        # Intersection found
        if region[x1, y1]:
            return True

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    # Check the final point
    if region[x2, y2]:
        return True

    # No intersection found
    return False
