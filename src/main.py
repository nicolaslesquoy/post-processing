# Standard Python libraries
import sys
import os

# Robust import of a TOML-parser
if sys.version_info < (3, 10):
    import tomli as tomllib
else:
    import tomllib
import pathlib

# Third party libraries
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import numpy as np
import cv2

# Local modules
sys.path.append(os.path.join(os.path.dirname(__file__), "../utils"))

from calibration import ImageCalibration as ic
from calibration import CenterCalibration as cc
from operations import FileOperations as fop

# Custom types
from custom_types import Path


# Configuration
with open("config.toml", "rb") as f:
    config = tomllib.load(f)

# config = working_directory("src")
# print(config)

# Path definitions
"""
    - path_to_calibration: Path to the folder containing the calibration images.
    - path_to_camera_calibration: Path to the folder containing the camera calibration images.
    - path_to_debug: Path to the folder containing the intermediary files used for debugging.
    - path_to_final: Path to the folder containing the final images.
"""
PATH_TO_CALIBRATION = pathlib.Path(config["paths"]["path_to_calibration"])
PATH_TO_DEBUG = pathlib.Path(config["paths"]["path_to_debug"])
PATH_TO_RAW = pathlib.Path(config["paths"]["path_to_raw"])
PATH_TO_FINAL = pathlib.Path(config["paths"]["path_to_final"])
# Positions of the calibration points on the test bench
CALIBRATION_POSITIONS = config["measures_calibration"]
# List of paths to the raw images
PATH_TO_RAW_IMAGES_FOLDER = [
    PATH_TO_RAW / "incidence_std",
    PATH_TO_RAW / "derapage_std",
    PATH_TO_RAW / "incidence_long",
    PATH_TO_RAW / "derapage_long",
    PATH_TO_RAW / "incidence_canards",
    PATH_TO_RAW / "derapage_canards",
]
PATH_TO_CALIBRATION_IMAGES = [
    PATH_TO_CALIBRATION / "image_calibration",
    PATH_TO_CALIBRATION / "center_calibration",
]
PATH_TO_ENDUIT = PATH_TO_RAW / "enduit"


class GlobalDriver:
    """This class is used to calibrate the images."""

    def __init__(
        self,
        path_to_calibration_folder: list[Path],
        path_to_raw_folder: list[Path],
        path_to_debug: Path,
        path_to_final: Path,
        calibration_positions: dict[str, list[np.float32]],
        calibrate_flag: bool = False,
        verify_flag: bool = False,
        analysis_flag: bool = False,
        graph_flag: bool = False,
        rotate: bool = True,
        resize: bool = False
    ) -> None:
        """This method is used to initialize the class.

        Parameters
        ----------
        path_to_calibration : list[Path]
            List of paths to the calibration images.
        path_to_raw : list[Path]
            List of paths to the raw images.
        path_to_debug : Path
            Path to the folder containing the intermediary files used for debugging.
        path_to_final : Path
            Path to the folder containing the final images.
        calibration_positions : dict[str, list[np.float32]]
            Dictionary containing the positions of the calibration points on the test bench.
        calibrate_flag : bool, optional
            Flag to launch the calibration process, by default False
        verify_flag : bool, optional
            Flag to launch the verification process, by default False
        analysis_flag : bool, optional
            Flag to launch the analysis process, by default False
        graph_flag : bool, optional
            Flag to launch the generation of the graphs, by default False
        """
        self.path_to_calibration_folder = path_to_calibration_folder
        self.path_to_raw_folder = path_to_raw_folder
        self.path_to_debug = path_to_debug
        self.path_to_final = path_to_final
        self.calibration_positions = calibration_positions
        self.calibrate_flag = calibrate_flag
        self.verify_flag = verify_flag
        self.analysis_flag = analysis_flag
        self.graph_flag = graph_flag
        self.rotate = rotate
        self.resize = resize

    def calibrate(self) -> bool:
        """This method is used to calibrate the images."""
        # Calibration
        if self.calibrate_flag:
            image_calibration_dataframe = cc.create_reference_dataframe(self.path_to_calibration_folder[0], self.calibration_positions)
            center_calibration_dataframe = ic.create_reference_dataframe(self.path_to_calibration_folder[1], self.calibration_positions)
            fop.save_dataframe_to_pickle(image_calibration_dataframe, self.path_to_debug / "image_calibration.pkl")
            fop.save_dataframe_to_pickle(center_calibration_dataframe, self.path_to_debug / "center_calibration.pkl")
            return True
        else:
            return False
        

