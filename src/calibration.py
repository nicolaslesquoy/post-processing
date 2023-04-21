# Standard Python libraries
import pathlib
import json
import tomli as tomllib  # Make this import robust for Python 3.11

# Third party libraries
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import scipy as sp
import pandas as pd
import cv2

# Local libraries
import operations


class Calibration:
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)

    # Send the click to the log file from the function 'onclick'.
    # send_click_to_log = config["parameters"]["send_onclick_to_log"] # TODO Implement this feature

    # Useful path definitions
    """
    - path_to_calibration: Path to the folder containing the calibration images.
    - path_to_camera_calibration: Path to the folder containing the camera calibration images.
    - path_to_debug: Path to the folder containing the intermediary files used for debugging.
    - path_to_logs: Path to the folder containing the log files. # TODO Use 'logging' librairy when ready.
    - path_to_calibration_log: Path to the log file containing the calibration logs.
    - path_to_intermediary: Path to the intermediary file containing the coordinates of the corners of the calibration zone for each image.
    """
    path_to_calibration = pathlib.Path(config["paths"]["path_to_calibration"])
    path_to_camera_calibration = pathlib.Path(config["paths"]["path_to_camera_calibration"])
    path_to_debug = pathlib.Path(config["paths"]["path_to_debug"])
    # path_to_logs = pathlib.Path(config["paths"]["path_to_logs"]) # TODO Use these files when implementing 'logging'
    # path_to_calibration_log = path_to_logs / "calibration.log"
    path_to_intermediary = path_to_debug / "intermediary.csv"

    # Cleaning functions

    def clear_intermediary_file(path_to_intermediary_file: pathlib.Path) -> bool:
        """Remove the intermediary file if it exists."""
        if path_to_intermediary_file.exists():
            path_to_intermediary_file.unlink()
            return True
        else:
            return False
        
    # Core functions

    def calibration(path_to_image: pathlib.Path, rotation: int = 180) -> None:
        """
        This function is used to calibrate the camera.
        It takes a path to an image as input and returns a dictionary with the coordinates of the corners of the image.
        """
        name = path_to_image.stem
        fig,ax = plt.subplots()
        img = np.asarray(Image.open(path_to_image))
        img = cv2.rotate(img, cv2.ROTATE_180)
        ax.imshow(img)

        def onclick(event):
            with open(Calibration.path_to_intermediary, "a") as f:
                f.write(f"{name},{event.xdata},{event.ydata}" + "\n")
            ax.plot(event.xdata, event.ydata, ".", c="b")
            fig.canvas.draw()
        
        cid = fig.canvas.mpl_connect("button_press_event", onclick)
        plt.show()

    def iterate(path_to_folder: pathlib.Path):
        """
        Iterate over the files in a folder and apply the calibration function to each of them.
        """
        for path in path_to_folder.iterdir():
            if path.is_file():
                Calibration.calibration(path)
        return None
        

    def create_reference_dataframe(path_to_intermediary: pathlib.Path):
        """
        Creates a reference file for the calibration and orders them in clockwise order.
        """
        df = pd.read_csv(
            path_to_intermediary, sep=",", header=None, index_col=0
        )
        result_dict = {}
        for image, data in df.groupby(0):
            points = []
            for index, row in data.iterrows():
                points.append((int(float(row[1])), int(float(row[2]))))
            result_dict[str(image)] = points
        return operations.Operations.order_corners_clockwise(result_dict)
    
    def create_calibration_json(result_dict: dict) -> None:
        """Create a json file with the coordinates of the corners of the image."""
        with open(Calibration.path_to_debug / "calibration.json", "w") as f:
            json.dump(result_dict, f, indent=4)
        return None

    def create_sanity_check(result_dict: dict) -> None:
        """Create a sanity check image of all founds coreners"""
        print("sanity check")
        for key in result_dict:
            points = np.array(list(map(list, result_dict[key])))
            x_points = points[:, [0]]
            y_points = points[:, [1]]
            plt.scatter(x_points, y_points, marker=".", label=f"{key}")
        plt.legend(loc="best", fontsize="small")
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.savefig(Calibration.path_to_debug / "sanity.jpg")
        return None
    
    def driver(iteration: bool = False):
        """Driver function for the calibration process."""
        if iteration:
            # Clearing the calibration file
            Calibration.clear_intermediary_file(Calibration.path_to_intermediary)
            # Iterating over all images
            Calibration.iterate(Calibration.path_to_calibration)

        # Creating the reference file
        result = Calibration.create_reference_dataframe(Calibration.path_to_intermediary)
        # Creating the json file
        Calibration.create_calibration_json(result)
        # Creating a sanity check image
        Calibration.create_sanity_check(result)
        return 0
    
class Interpolation:
    def __init__(self, path_to_calibration: pathlib.Path = Calibration.path_to_debug / "calibration.json"):
        with open(path_to_calibration, "r") as f:
            self.calibration = json.load(f)
    
    def interpolate(result_dict: dict) -> tuple:
        """
        Interpolate the coordinates of a point in a given image.
        """
        for key in result_dict:
            points = np.array(list(map(list, result_dict[key])))
            x_points = points[:, [0]]
            y_points = points[:, [1]]
            plt.scatter(x_points, y_points, marker=".", label=f"{key}")


    
if __name__ == "__main__":
    Calibration.driver(iteration=True)

