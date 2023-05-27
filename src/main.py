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
from operations import PlotOperations as pop
from analysis import Analysis as an
from drawing import Drawing as dr

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
            center_calibration_dataframe = None  # TODO : implement this feature
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
                        final_dataframe,
                        self.path_to_debug / f"result_dataframe_{path.stem}.pkl",
                    )
            return True
        else:
            return False

    def draw_graph1(self, result_dict) -> bool:
        """This method is used to draw the first graph.
        evolution config standard incidence"""
        if self.graph_flag:
            dx, dy = 0.01, 0.01
            print("Drawing graph 1")
            color = {"15": "red", "20": "green", "25": "blue"}
            # Load the dataframes
            for key in result_dict.keys():
                for point in result_dict[key]:
                    plt.scatter(
                        point["coordinates"][0] * dx,
                        point["coordinates"][1] * dy,
                        color=color[key],
                    )
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
        analysis_flag=True,
        graph_flag=True,
        rotate=True,
        resize=False,
    )

    # Launch the calibration
    # global_driver.calibrate(calibrate_images=True, calibrate_center=False)
    global_driver.analyse(exclude_folder=[1, 2, 3, 4, 5])
    # center_dict = cc.create_center_dataframe(PATH_TO_REFERENCE)
    # incidence_std_dataframe = fop.load_pickle_to_dataframe(PATH_TO_DEBUG / "result_dataframe_incidence_std.pkl")
    # incidence_std_dict = pop.load_points(incidence_std_dataframe, True, False)
    # incidence_std_dict = pop.extract_points_from_dict(incidence_std_dict, ["clean", "clean_fuselage", "clean_smaller_vortices"])
    # incidence_std_dict = pop.prepare_points(incidence_std_dict, center_dict, CALIBRATION_POSITIONS)
    # dr.draw_graph_X("Evolution de la position du tourbillon \n en fonction de l'incidence en configuration standard", "incidence_std", PATH_TO_FINAL, incidence_std_dict, CALIBRATION_POSITIONS, {"15": "red", "20": "green", "25": "blue"}, dx=0.01, yerr=0)
    # dr.draw_graph_XY("Evolution de la position du tourbillon \n en fonction de l'incidence en configuration standard", "incidence_std", PATH_TO_FINAL, incidence_std_dict, CALIBRATION_POSITIONS, {"15": "red", "20": "green", "25": "blue"}, dx=0.01, dy=0.01, xerr=0, yerr=0)
    # derapage_std_dataframe = fop.load_pickle_to_dataframe(PATH_TO_DEBUG / "result_dataframe_derapage_std.pkl")
    # derapage_std_dict = pop.load_points(derapage_std_dataframe, False, True)
    # derapage_std_dict = pop.extract_points_from_dict(derapage_std_dict, ["clean", "clean_fuselage", "clean_smaller_vortices"])
    # derapage_std_dict = pop.prepare_points(derapage_std_dict, center_dict, CALIBRATION_POSITIONS)
    # derapage_std_dict_merge = {
    #     "0": incidence_std_dict["20"],
    #     "5": derapage_std_dict["5"],
    #     "10": derapage_std_dict["10"],
    # }
    # dr.draw_graph_X("Evolution de la position du tourbillon \n en fonction du dérapage en configuration standard", "derapage_std", PATH_TO_FINAL, derapage_std_dict_merge, CALIBRATION_POSITIONS, {"0": "red", "5": "green", "10": "blue"}, dx=0.01, yerr=0)
    # dr.draw_graph_XY("Evolution de la position du tourbillon \n en fonction du dérapage en configuration standard", "derapage_std", PATH_TO_FINAL, derapage_std_dict_merge, CALIBRATION_POSITIONS, {"0": "red", "5": "green", "10": "blue"}, dx=0.01, dy=0.01, xerr=0, yerr=0)
    # incidence_long_dataframe = fop.load_pickle_to_dataframe(PATH_TO_DEBUG / "result_dataframe_incidence_long.pkl")
    # incidence_long_dict = pop.load_points(incidence_long_dataframe, True, False)
    # incidence_long_dict = pop.extract_points_from_dict(incidence_long_dict, ["clean", "clean_fuselage", "clean_smaller_vortices"])
    # incidence_long_dict = pop.prepare_points(incidence_long_dict, center_dict, CALIBRATION_POSITIONS)
    # incidence_canards_dataframe = fop.load_pickle_to_dataframe(PATH_TO_DEBUG / "result_dataframe_incidence_canards.pkl")
    # incidence_canards_dict = pop.load_points(incidence_canards_dataframe, True, False)
    # incidence_canards_dict = pop.extract_points_from_dict(incidence_canards_dict, ["clean", "clean_fuselage", "clean_smaller_vortices"])
    # incidence_canards_dict = pop.prepare_points(incidence_canards_dict, center_dict, CALIBRATION_POSITIONS)
    # incidence_merge = {
    #     "std": incidence_std_dict["20"],
    #     "long": incidence_long_dict["20"],
    #     "canards": incidence_canards_dict["20"],
    # }
    # dr.draw_graph_X("Evolution de la position du tourbillon \n en fonction de la configuration", "config", PATH_TO_FINAL, incidence_merge, CALIBRATION_POSITIONS, {"std": "red", "long": "green", "canards": "blue"}, dx=0.01, yerr=0)
    # dr.draw_graph_XY("Evolution de la position du tourbillon \n en fonction de la configuration", "config", PATH_TO_FINAL, incidence_merge, CALIBRATION_POSITIONS, {"std": "red", "long": "green", "canards": "blue"}, dx=0.01, dy=0.01, xerr=0, yerr=0)
    # # dataframe_1 = fop.load_pickle_to_dataframe(
    # #     PATH_TO_DEBUG / "result_dataframe_derapage_std.pkl"
    # # )
    # # result_dict_1 = pop.load_points(dataframe_1, False, True)
    # # # print(result_dict)
    # # result_dict_1 = pop.extract_points_from_dict(result_dict_1, ["clean", "clean_fuselage", "clean_smaller_vortices"])
    # # result_dict_1 = pop.prepare_points(result_dict_1, center_dict, CALIBRATION_POSITIONS)
    # # dataframe_2= fop.load_pickle_to_dataframe(PATH_TO_DEBUG / "result_dataframe_incidence_std.pkl")
    # # result_dict_2 = pop.load_points(dataframe_2, True, False)
    # # result_dict_2 = pop.extract_points_from_dict(result_dict_2, ["clean", "clean_fuselage", "clean_smaller_vortices"])
    # # result_dict_2 = pop.prepare_points(result_dict_2, center_dict, CALIBRATION_POSITIONS)
    # # result_dict = {
    # #     "0": result_dict_2["20"],
    # #     "5": result_dict_1["5"],
    # #     "10": result_dict_1["10"],
    # # }
    # # dr.draw_graph_X("Evolution de la position du tourbillon \n dans le plan de l'aile", "test", PATH_TO_FINAL,result_dict, CALIBRATION_POSITIONS, {"0": "red", "5": "green", "10": "blue"}, 0.01, 0.5)
    # # dr.draw_graph_XY("Evolution de la position du tourbillon", "test", PATH_TO_FINAL,result_dict, CALIBRATION_POSITIONS, {"0": "red", "5": "green", "10": "blue"}, 0.01, 0.01, 0.5, 0.5)

