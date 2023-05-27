# Standard Python libraries
from functools import reduce
import operator

# Third-party libraries
import numpy as np
from PIL import Image
import pandas as pd

# Local packages
from custom_types import Path, Dataframe, DictPoints, Coordinates, NumpyArray


class CalibrationOperations:
    """
    Class containing the operations used for the calibration.
    """
    def order_points_clockwise(coords: Coordinates) -> Coordinates:
        """Order the points in a clockwise fashion.

        Parameters
        ----------
        coords : Coordinates
            List of coordinates

        Returns
        -------
        Coordinates
            List of coordinates ordered in clockwise order.
        """
        center = tuple(map(operator.truediv, reduce(
            lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
        return sorted(coords, key=lambda coord: (-135 - np.degrees(np.arctan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)

    def create_destination_points(Nx: int, Ny: int, dx: int, dy: int) -> NumpyArray:
        """Create the destination points for the unwarping.

        Parameters
        ----------
        Nx : int
            Number of pixels in the x direction.
        Ny : int
            Number of pixels in the y direction.
        dx : int
            Length of the target zone along the x direction.
        dy : int
            Length of the target zone along the y direction.

        Returns
        -------
        NumpyArray
            Array containing the destination points as numpy.float32.
        """
        #! TODO : put dx and dy as parameters in config.toml
        center = [Nx / 2, Ny / 2]
        p1, p2, p3, p4 = [center[0] + dx, center[1] + dy], [center[0] - dx, center[1] +
                                                            dy], [center[0] - dx, center[1] - dy], [center[0] + dx, center[1] - dy]
        return np.array(CalibrationOperations.order_points_clockwise([p1, p2, p3, p4]), dtype=np.float32)
    
    def get_conversion(Nx, Ny):
        reference = CalibrationOperations.create_destination_points(Nx, Ny, 500, 500)
        # print(reference)
        delta_nx = abs(reference[1][0] - reference[0][0])
        delta_ny = abs(reference[2][1] - reference[0][1])
        delta_x = 10  # cm
        delta_y = 10  # cm
        dx = delta_x / delta_nx
        dy = delta_y / delta_ny
        return [dx, dy]

    def read_file_name(path: Path) -> list[str]:
        """Read the file name from the path to ouput its information.
        The file name must follow the convention <AOA>-<DA>-<reference>.<extension>.

        Parameters
        ----------
        path : Path
            Path to the file.

        Returns
        -------
        list[str]
            List containing the information from the file name with the following order:
            - AOA: angle of attack
            - DA: drift angle
            - reference: reference point
        """
        name = path.stem
        return name.split("-")


class FileOperations:

    def open_image_as_array(path_to_file: Path) -> NumpyArray:
        """
        Opens an image as a numpy array.
        """
        return np.asanyarray(Image.open(path_to_file))

    def save_dict_as_dataframe(dictionary: DictPoints) -> Dataframe:
        """Save the dictionary as a dataframe."""
        keys = list(dictionary.keys())
        keys.sort()
        data = {key: dictionary[key] for key in keys}
        dataframe = pd.DataFrame.from_dict(data, orient="index")
        return dataframe

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
    
    def get_middle(p1: list[float], p2: list[float]) -> list[float]:
        """Get the middle point between two points.

        Parameters
        ----------
        p1 : list[float]
            First point.
        p2 : list[float]
            Second point.

        Returns
        -------
        list[float]
            Middle point.
        """
        return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]

    def get_distance(p1: list[float], p2: list[float]) -> float:
        """Get the distance between two points.

        Parameters
        ----------
        p1 : list[float]
            First point.
        p2 : list[float]
            Second point.

        Returns
        -------
        float
            Distance between the two points.
        """
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def get_coordinates_from_point(ref: list[float], point: list[float]):
        """Get the coordinates of a point from a reference point.

        Parameters
        ----------
        ref : list[float]
            Reference point.
        point : list[float]
            Point to get the coordinates from.

        Returns
        -------
        list[float]
            Coordinates of the point from the reference point.
        """
        return [point[0] - ref[0], abs(point[1] - ref[1])] # y is inverted
    
    def get_conversion(Nx,Ny):
        dst_points = CalibrationOperations.create_destination_points(Nx, Ny, 500, 500)
        dx = dst_points[0][0] - dst_points[1][0]
        dy = dst_points[0][1] - dst_points[2][1]
        return dx/10, dy/10



class ImageOperations:

    def convert_coordinates(coordinates: Coordinates, dx: int, dy: int):
        return [[x - dx, y - dy] for x, y in coordinates]


class LineBuilder(object):

    def __init__(self, line):
        canvas = line.figure.canvas
        self.canvas = canvas
        self.line = line
        self.axes = line.axes
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.epsilon = 50
    
        self.ind = None
    
        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect('button_release_event', self.button_release_callback)
        canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)

    def get_ind(self, event):
        x = np.array(self.line.get_xdata())
        y = np.array(self.line.get_ydata())
        d = np.sqrt((x-event.xdata)**2 + (y - event.ydata)**2)
        if min(d) > self.epsilon:
            return None
        if d[0] < d[1]:
            return 0
        else:
            return 1

    def button_press_callback(self, event):
        if event.button != 1:
            return
        self.ind = self.get_ind(event)
    
        self.line.set_animated(True)
        self.canvas.draw()
        self.background = self.canvas.copy_from_bbox(self.line.axes.bbox)
    
        self.axes.draw_artist(self.line)
        self.canvas.blit(self.axes.bbox)

    def button_release_callback(self, event):
        if event.button != 1:
            return
        self.ind = None
        self.line.set_animated(False)
        self.background = None
        self.line.figure.canvas.draw()

    def motion_notify_callback(self, event):
        if event.inaxes != self.line.axes:
            return
        if event.button != 1:
            return
        if self.ind is None:
            return
        self.xs[self.ind] = event.xdata
        self.ys[self.ind] = event.ydata
        self.line.set_data(self.xs, self.ys)
    
        self.canvas.restore_region(self.background)
        self.axes.draw_artist(self.line)
        self.canvas.blit(self.axes.bbox)
