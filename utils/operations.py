# Standard Python libraries
from functools import reduce
import operator
import pathlib

# Third-party libraries
import numpy as np
from PIL import Image
import pandas as pd

# Local packages
from custom_types import Path, Dataframe, DictPoints, Coordinates, NumpyArray

class CalibrationOperations:

    def order_points_clockwise(coords: Coordinates) -> Coordinates:
        center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
        return sorted(coords, key=lambda coord: (-135 - np.degrees(np.arctan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)

    def create_destination_points(Nx: int, Ny: int, dx: int, dy: int) -> NumpyArray:
        center = [Nx / 2, Ny / 2]
        p1,p2,p3,p4 = [center[0] + dx, center[1] + dy], [center[0] - dx, center[1] + dy], [center[0] - dx, center[1] - dy], [center[0] + dx, center[1] - dy]
        return np.array(CalibrationOperations.order_points_clockwise([p1, p2, p3, p4]), dtype=np.float32)
    
class FileOperations:

    def open_image_as_array(path_to_file: Path) -> NumpyArray:
        """
        Opens an image as a numpy array.
        """
        return np.asanyarray(Image.open(path_to_file))
    
    def save_dataframe_to_pickle(dataframe: Dataframe, path_to_file: Path) -> None:
        """Save the dataframe to a pickle file."""
        dataframe.to_pickle(path_to_file)
        return None
    
    def load_pickle_to_dataframe(path_to_file: Path) -> Dataframe:
        """Load the dataframe from a pickle file."""
        dataframe = pd.read_pickle(path_to_file)
        return dataframe
    
class MathOperations:

    def compute_r2(xdata, ydata, func, popt):
        residuals = ydata - func(xdata, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2
    
class ImageOperations:

# def translate_points(points, a, b):
#     translated_points = []
#     for point in points:
#         x = point[0] - a
#         y = point[1] - b
#         translated_points.append([x, y])
#     return translated_points
    def convert_coordinates(coordinates: Coordinates, dx: int, dy: int):
        return [[x - dx, y - dy] for x, y in coordinates]

    def refine_position(x: float, y: float, dx: int, dy: int):
        pass