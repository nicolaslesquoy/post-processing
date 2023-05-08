# Standard Python libraries
import os
import contextlib
import pathlib
import tomllib

# Third party libraries

# Local libraries
import operations

# Constants

# Configuration

@contextlib.contextmanager
def working_directory(path):
    """Changes working directory and returns to previous on exit."""
    prev_cwd = pathlib.Path.cwd()
    os.chdir(path)
    try:
        print("1")
        with open("../config.toml", "rb") as f:
            config = tomllib.load(f)
        yield config
    finally:
        print("2")
        os.chdir(prev_cwd)

config = working_directory("src")
print(config)

# Path definitions
"""
    - path_to_calibration: Path to the folder containing the calibration images.
    - path_to_camera_calibration: Path to the folder containing the camera calibration images.
    - path_to_debug: Path to the folder containing the intermediary files used for debugging.
    - path_to_final: Path to the folder containing the final images.
"""
path_to_calibration = pathlib.Path(config["paths"]["path_to_calibration"])
path_to_camera_calibration = pathlib.Path(config["paths"]["path_to_camera_calibration"])
path_to_debug = pathlib.Path(config["paths"]["path_to_debug"])
path_to_raw = pathlib.Path(config["paths"]["path_to_raw"])
path_to_final = pathlib.Path(config["paths"]["path_to_final"])
# Positions of the calibration points on the test bench
calibration_positions = config["measures_calibration"]



class Calibration:
    pass

class CameraCalibration:
    pass