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

# Local libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils')) 
import operations
import calibration

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

if __name__ == "__main__":

    test = calibration.ImageCalibration(PATH_TO_CALIBRATION, PATH_TO_DEBUG / "image_calibration.pkl", CALIBRATION_POSITIONS)
    dataframe = calibration.ImageCalibration.create_reference_dataframe(test)
    operations.FileOperations.save_dataframe_to_pickle(dataframe, PATH_TO_DEBUG / "image_calibration.pkl")
    df = operations.FileOperations.load_pickle_to_dataframe(PATH_TO_DEBUG / "image_calibration.pkl")
    df.to_string("test.txt")
    # test_verif = calibration.ImageCalibrationVerification(PATH_TO_CALIBRATION, PATH_TO_DEBUG / "image_calibration.pkl", CALIBRATION_POSITIONS, [])
    # print(calibration.ImageCalibrationVerification.check_reference_points(test_verif))
