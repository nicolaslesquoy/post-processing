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
from operations import CalibrationOperations as cop
from operations import FileOperations as fop
from operations import MathOperations as mop
from analysis import Analysis as an

# Custom types
from custom_types import Path, Dataframe


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
PATH_TO_REFERENCE = pathlib.Path(config["paths"]["path_to_reference"])
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
        path_to_reference: Path,
        path_to_final: Path,
        calibration_positions: dict[str, list[np.float32]],
        calibrate_flag: bool = False,
        verify_flag: bool = False,
        analysis_flag: bool = False,
        graph_flag: bool = False,
        rotate: bool = True,
        resize: bool = False,
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
        self.path_to_reference = path_to_reference
        self.path_to_final = path_to_final
        self.calibration_positions = calibration_positions
        self.calibrate_flag = calibrate_flag
        self.verify_flag = verify_flag
        self.analysis_flag = analysis_flag
        self.graph_flag = graph_flag
        self.rotate = rotate
        self.resize = resize

    def calibrate(
        self, calibrate_center: bool = False, calibrate_images: bool = False
    ) -> bool:
        """This method is used to calibrate the images."""
        # Calibration
        if self.calibrate_flag and (calibrate_center or calibrate_images):
            if calibrate_images:
                print("Image Calibration")
                image_calibration_dataframe = ic.create_reference_dataframe(
                    self.path_to_calibration_folder[0],
                    self.calibration_positions,
                    resize=self.resize,
                )
                fop.save_dataframe_to_pickle(
                    image_calibration_dataframe,
                    self.path_to_debug / "image_calibration.pkl",
                )
            if calibrate_center:
                print("Center Calibration")
                calibration_dataframe = fop.load_pickle_to_dataframe(
                    self.path_to_debug / "image_calibration.pkl"
                )
                center_calibration_dataframe = cc.create_reference_dataframe(
                    self.path_to_calibration_folder[1],
                    calibration_dataframe,
                    resize=self.resize,
                    rotate=self.rotate,
                )
                fop.save_dataframe_to_pickle(
                    center_calibration_dataframe,
                    self.path_to_debug / "center_calibration.pkl",
                )
            return True
        else:
            return False

    def analyse(self, exclude_folder: list[int] = []) -> bool:
        """This method is used to analyse the images."""
        # Analysis
        if self.analysis_flag:
            print("Analysis")
            # Load the calibration dataframes
            image_calibration_dataframe = fop.load_pickle_to_dataframe(
                self.path_to_debug / "image_calibration.pkl"
            )
            center_calibration_dataframe = None # TODO : implement this feature
            # Create the final dataframes
            for path in self.path_to_raw_folder:
                if self.path_to_raw_folder.index(path) not in exclude_folder:
                    print(f"Processing {path}")
                    final_dataframe = an.driver(
                        path,
                        image_calibration_dataframe,
                        center_calibration_dataframe,
                        resize=self.resize,
                    )
                    fop.save_dataframe_to_pickle(
                        final_dataframe, self.path_to_debug / f"result_dataframe_{path.stem}.pkl"
                    )
            return True
        else:
            return False
    
    def load_points(dataframe: Dataframe, incidence: bool, derapage: bool) -> bool: 
        """This method returns a dictionary containing the points of the dataframe separated by incidence or derapage."""
        list_index = dataframe.index
        if incidence:
            list_incidence = sorted(list(set([list_index[i].split("-")[0] for i in range(len(list_index))])))
            result_dict = {key : [] for key in list_incidence} # initialize the dictionary
            for row in dataframe.iterrows():
                result_dict[row[0].split("-")[0]].append(row[1].to_dict())
            return result_dict
        if derapage:
            list_derapage = sorted(list(set([list_index[i].split("-")[1] for i in range(len(list_index))])))
            result_dict = {key : [] for key in list_derapage} # initialize the dictionary
            for row in dataframe.iterrows():
                result_dict[row[0].split("-")[1]].append(row[1].to_dict())
            return result_dict
        
    def clean_dict(result_dict: dict) -> dict:
        """This method returns a copy of the dictionary cleaned."""
        copy_dict = result_dict.copy()
        for key in copy_dict.keys():
            for point in copy_dict[key]:
                if point["center"] == None:
                    copy_dict[key].remove(point)
        return copy_dict
    
    def clean_dict_fuselage(result_dict: dict) -> dict:
        """This method returns a copy of the dictionary cleaned."""
        copy_dict = result_dict.copy()
        for key in copy_dict.keys():
            for point in copy_dict[key]:
                point.pop("rectangle_fuselage")
        return copy_dict
    
    def clean_dict_smaller_vortices(result_dict: dict) -> dict:
        """This method returns a copy of the dictionary cleaned."""
        copy_dict = result_dict.copy()
        for key in copy_dict.keys():
            for point in copy_dict[key]:
                if point["rectangle_stable"] != None and len(point["rectangle_stable"]) > 1:
                    point["rectangle_stable"].remove(point["rectangle_stable"][0])
        return copy_dict
    
    def extract_points_from_dict(result_dict: dict, list_type: str):
        """This method returns a copy of the dictionary cleaned."""
        copy_dict = result_dict.copy()
        for type in list_type:
            if type == "clean":
                copy_dict = GlobalDriver.clean_dict(copy_dict)
            if type == "clean_fuselage":
                copy_dict = GlobalDriver.clean_dict_fuselage(copy_dict)
            if type == "clean_smaller_vortices":
                copy_dict = GlobalDriver.clean_dict_smaller_vortices(copy_dict)
        return copy_dict
        
    def prepare_points(result_dict: dict, center_dict: dict):
        filtered_dict = {}
        for key in result_dict.keys():
            list_of_points = []
            for point in result_dict[key]:
                name = point["name"]
                keys_list = list(point.keys())
                if point["rectangle_stable"] != None:
                    for rectangle in point["rectangle_stable"]:
                        if abs(rectangle["a"][0] - rectangle["b"][0]) > 1e-3:
                            list_of_points.append({"label": f"{name}/rectangle_stable", "coordinates": mop.get_coordinates_from_point(center_dict[name],mop.get_middle(rectangle["a"], rectangle["b"]))})
                            print(1)
                if "rectangle_fuselage" in keys_list and point["rectangle_fuselage"] != None:
                    for rectangle in point["rectangle_fuselage"]:
                        if abs(rectangle["a"][0] - rectangle["b"][0]) > 1e-3:
                            list_of_points.append({"label": f"{name}/rectangle_fuselage", "coordinates": mop.get_coordinates_from_point(center_dict[name],mop.get_middle(rectangle["a"], rectangle["b"]))})
                            print(2)
                if point["rectangle_unstable"] != None:
                    for rectangle in point["rectangle_unstable"]:
                        if abs(rectangle["a"][0] - rectangle["b"][0]) > 1e-3:
                            list_of_points.append({"label": f"{name}/rectangle_unstable", "coordinates": mop.get_coordinates_from_point(center_dict[name],mop.get_middle(rectangle["a"], rectangle["b"]))})
                            print(3)
            filtered_dict[key] = list_of_points
        return filtered_dict

    def draw_graph1(self, result_dict) -> bool:
        """This method is used to draw the first graph.
        evolution config standard incidence"""
        if self.graph_flag:
            dx, dy = 0.01, 0.01
            print("Drawing graph 1")
            color = {"15": "red","20": "green","25": "blue"}
            # Load the dataframes
            for key in result_dict.keys():
                for point in result_dict[key]:
                    plt.scatter(point["coordinates"][0]*dx, point["coordinates"][1]*dy, color=color[key])
            plt.savefig(f"{self.path_to_debug}-{key}.png")
            plt.clf()
            return True
        else:
            return False

if __name__ == "__main__":
    # Create the class
    global_driver = GlobalDriver(
        path_to_calibration_folder=PATH_TO_CALIBRATION_IMAGES,
        path_to_raw_folder=PATH_TO_RAW_IMAGES_FOLDER,
        path_to_debug=PATH_TO_DEBUG,
        path_to_reference=PATH_TO_REFERENCE,
        path_to_final=PATH_TO_FINAL,
        calibration_positions=CALIBRATION_POSITIONS,
        calibrate_flag=False,
        verify_flag=False,
        analysis_flag=False,
        graph_flag=True,
        rotate=True,
        resize=False,
    )

    # Launch the calibration
    global_driver.calibrate(calibrate_images=True, calibrate_center=False)
    global_driver.analyse(exclude_folder=[1,2,3,4,5])
    center_dict = cc.create_center_dataframe(PATH_TO_REFERENCE)
    dataframe = fop.load_pickle_to_dataframe(PATH_TO_DEBUG / "result_dataframe_incidence_std.pkl")
    result_dict = GlobalDriver.load_points(dataframe, True, False)
    result_dict = GlobalDriver.extract_points_from_dict(result_dict, ["clean", "clean_fuselage", "clean_smaller_vortices"])
    result_dict = GlobalDriver.prepare_points(result_dict, center_dict)
    global_driver.draw_graph1(result_dict)
