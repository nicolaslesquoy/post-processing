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