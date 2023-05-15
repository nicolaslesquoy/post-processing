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

# Local libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils')) 
from operations import FileOperations as fop
from operations import CalibrationOperations as cop
import calibration
from warp import Warp

# Custom types
from custom_types import Path, Dataframe, NumpyArray


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
PATH_TO_CAMERA_CALIBRATION = pathlib.Path(
    config["paths"]["path_to_camera_calibration"])
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
PATH_TO_ENDUIT = PATH_TO_RAW / "enduit"

class Analysis:
    def draw_rectangle(name: str, img: NumpyArray):
        fig, ax = plt.subplots()
        ax.imshow(img)
        var_data = []

        def select_callback(eclick, erelease):
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            rect = plt.Rectangle(
                (min(x1, x2), min(y1, y2)),
                np.abs(x1 - x2),
                np.abs(y1 - y2),
                facecolor="none",
                edgecolor="red",
                linewidth=2,
            )
            ax.add_patch(rect)
            nonlocal var_data
            var_data.append({"a": [x1, y1], "b": [x2, y2]})
            # print("({:.3f}, {:.3f}) --> ({:.3f}, {:.3f})".format(x1, y1, x2, y2))

        plt.title(name + " - Select the rectangle")
        rs = RectangleSelector(
            ax,
            select_callback,
            useblit=False,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
        )

        plt.show()

        if len(var_data) == 0:
            return None
        else:
            return var_data

    def get_center(name: str, img: NumpyArray):
        fig, ax = plt.subplots()
        ax.imshow(img)
        # Variable declaration
        var_data = []

        def onclick(event):
            nonlocal var_data
            var_data.append([event.xdata, event.ydata])
            ax.plot(event.xdata, event.ydata, ".", c="b")
            fig.canvas.draw()

        plt.title(name + " - Select the center")
        cid = fig.canvas.mpl_connect("button_press_event", onclick)
        plt.show()
        if len(var_data) == 0:
            return None
        else:
            return var_data[0]

    def driver(path_to_folder: Path, calibration_dataframe: Dataframe) -> Dataframe:
        """Driver function for the analysis process."""
        result_dict = {}
        for path_to_file in path_to_folder.glob("*.jpg"):
            warped = Warp.warp(path_to_file, calibration_dataframe)
            result_int = {}
            try:
                name = path_to_file.stem + "-" + path_to_folder.stem
                result_int["name"] = name
                result_int["rectangle"] = Analysis.draw_rectangle(name, warped)
                result_int["center"] = Analysis.get_center(name, warped)
                result_dict[name] = result_int
            except:
                pass
        return fop.save_dict_as_dataframe(result_dict)
    
class GlobalDriver:
    pass


if __name__ == "__main__":

    # calibration_init = calibration.ImageCalibration(PATH_TO_CALIBRATION, PATH_TO_DEBUG / "image_calibration.pkl", CALIBRATION_POSITIONS)
    # calibration_dataframe = calibration.ImageCalibration.create_reference_dataframe(calibration_init)
    # fop.save_dataframe_to_pickle(calibration_dataframe, calibration_init.path_to_output_file)
    df = fop.load_pickle_to_dataframe(PATH_TO_DEBUG / "image_calibration.pkl")
    # df.to_string(PATH_TO_DEBUG / "image_calibration.txt")
    # # test_verif = calibration.ImageCalibrationVerification(PATH_TO_CALIBRATION, PATH_TO_DEBUG / "image_calibration.pkl", CALIBRATION_POSITIONS, [])
    # # print(calibration.ImageCalibrationVerification.check_reference_points(test_verif))
    # Analysis.driver(PATH_TO_RAW_IMAGES_FOLDER[0], df)