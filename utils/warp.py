# Standard Python libraries

# Third party libraries
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import numpy as np

# Local packages
from operations import CalibrationOperations as cop
from operations import FileOperations as fop
from operations import ImageOperations as iop
from operations import MathOperations as mop

# Custom types
from custom_types import Path, Dataframe

class Warp:
    def __init__(self, path_to_folder: Path, calibration_dataframe: Dataframe) -> None:
        self.path_to_folder = path_to_folder
        self.calibration_dataframe = calibration_dataframe
    
    def __repr__(self) -> str:
        return f"Warp(path_to_folder={self.path_to_folder}, calibration_dataframe={self.calibration_dataframe})"
    
    def __str__(self) -> str:
        return self.__repr__
    
    def get_reference_points(image_name: str, calibration_dataframe: Dataframe):
        """Get the reference points from the dataframe."""
        try:
            reference = calibration_dataframe[calibration_dataframe["name"] == image_name].to_dict(orient="records")[0]
            return np.array([reference[0], reference[1], reference[2], reference[3]], dtype=np.float32)
        except Exception as e:
            print(e)
            return None
    
    def warp(path_to_image: Path, calibration_dataframe: Dataframe, rotate: bool = True, resize: bool = True):
        """Warp the image."""
        img = cv2.imread(str(path_to_image))
        if resize:
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        if rotate:
            img = cv2.rotate(img, cv2.ROTATE_180)
        dst_points = cop.create_destination_points(img.shape[1], img.shape[0], 500, 500)
        label = cop.read_file_name(path_to_image)
        ref_points = Warp.get_reference_points(label[2], calibration_dataframe)
        if ref_points is None:
            return None
        else:
            ref_points = np.array(ref_points, dtype=np.float32)
            M = cv2.getPerspectiveTransform(ref_points, dst_points)
            warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
            return warped