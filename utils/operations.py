from functools import reduce
import operator
import numpy as np
import pathlib
from PIL import Image

class CalibrationOperations:

    def order_points_clockwise(coords):
        center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
        return sorted(coords, key=lambda coord: (-135 - np.degrees(np.arctan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)

    def create_destination_points(Nx: int, Ny: int, dx: int, dy: int):
        center = [Nx / 2, Ny / 2]
        p1,p2,p3,p4 = [center[0] + dx, center[1] + dy], [center[0] - dx, center[1] + dy], [center[0] - dx, center[1] - dy], [center[0] + dx, center[1] - dy]
        return np.array(CalibrationOperations.order_points_clockwise([p1, p2, p3, p4]), dtype=np.float32)
    
class FileOperations:

    def open_image_as_array(path_to_file: pathlib.Path):
        """
        Opens an image as a numpy array.
        """
        return np.asanyarray(Image.open(path_to_file))