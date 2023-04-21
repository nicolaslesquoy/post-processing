import pathlib

import numpy as np

class Operations:
    """Common operations between the different modules."""

    # Geometry
    def centeroidnp(arr):
        """Finds the centroids of a numpy array of points."""
        length = arr.shape[0]
        sum_x = np.sum(arr[:, 0])
        sum_y = np.sum(arr[:, 1])
        return sum_x/length, sum_y/length
    
    def order_corners_clockwise(points):
        """
        Given a list of points representing the corners of a square, returns a new
        list of these points ordered clockwise.
        """
        if len(points) != 4:
            raise ValueError("The input should contain exactly four points")
        center = Operations.centeroidnp(np.array(points))
        points.sort(key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))
        return points
    
    # File operations

    def get_file_name(path: pathlib.Path):
        """Returns the file name of a path."""
        return ("" if (p := path).is_dir() else p.name) == ""