# Standard librairies
import os
import json
from functools import reduce
import operator
from pathlib import Path
import logging
# Make this import robust for Python 3.11
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

# Third party librairies
import matplotlib.pyplot as plt
from matplotlib import transforms
from PIL import Image
import numpy as np
import pandas as pd

class Calibration:
    # TODO Move to 'pathlib' librairy + implement logging in all functions
    with open("config.toml","rb") as f:
        config = tomllib.load(f)

    send_click_to_log = config["parameters"]["send_onclick_to_log"]

    path_to_calibration = Path(config["paths"]["path_to_calibration"])
    path_to_camera_calibration = Path(config["paths"]["path_to_camera_calibration"])
    path_to_debug = Path(config["paths"]["path_to_debug"])
    path_to_logs = Path(config["paths"]["path_to_logs"])
    path_to_calibration_log = path_to_logs / "calibration.log"
    path_to_intermediary = path_to_debug / "intermediary.csv"

    logging.basicConfig(filename=path_to_calibration_log,format='%(asctime)s - %(message)s',datefmt='%d-%b-%y %H:%M:%S',level=logging.DEBUG)

    def clear_calibration_file(path_to_calibration_file: Path) -> bool:
        """Remove the calibration file if it exists."""
        if os.path.exists(path_to_calibration_file):
            os.remove(path_to_calibration_file)
            return True
        else:
            return False
    
    def clear_log_file(path_to_log_file: Path) -> bool:
        """Remove the calibration file if it exists."""
        if os.path.exists(path_to_log_file):
            os.remove(path_to_log_file)
            return True
        else:
            return False

    def calibration(path_to_image: Path,rotation: int = 180):
        """
        This function is used to calibrate the camera.
        It takes a path to an image as input and returns a dictionary with the coordinates of the corners of the image.
        """
        try:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            img = np.asarray(Image.open(path_to_image))
            name = os.path.basename(path_to_image)
            print(name)
            tr = transforms.Affine2D().rotate_deg(rotation)
            ax.imshow(img,transforms=tr)
            def onclick(event):
                with open(Calibration.path_to_intermediary,"a") as f:
                    f.write(f"{name},{event.xdata},{event.ydata}" + "\n")
                    f.close()
                # Logging to calibration log file
                logging.debug('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %(event.button, event.x, event.y, event.xdata, event.ydata))
                plt.plot(event.xdata, event.ydata, '.',c='b')
                fig.canvas.draw()
            cid = fig.canvas.mpl_connect('button_press_event', onclick)
            plt.show()
        except Exception as e:
            logging.exception("Exception occurred")
        return 0
    
    def iterate(path_to_folder):
        """
        Iterate over all elements in the folder
        TODO Migrate to 'pathlib' module instead of os
        """
        for filename in os.listdir(path_to_folder):
            f = os.path.join(path_to_folder, filename)
            # checking if it is a file
            if os.path.isfile(f):
                Calibration.calibration(f)
        return None
    
    def create_reference_file(path_to_calibration_file):
        """
        Create a reference file for the calibration
        """
        df = pd.read_csv(Calibration.path_to_intermediary,sep=",",header=None,index_col=0)
        result_dict = {}
        for image,data in df.groupby(0):
            points = []
            for index,row in data.iterrows():
                points.append((int(float(row[1])),int(float(row[2]))))
            result_dict[image.strip('.jpg')] = points
        return result_dict

    def order_corner_clockwise(points_dict):
        """Orders all the found corners in clockwise order to be able to use them in the perspective transform with consistent results."""
        new_dict = {}
        for key in points_dict:
            coords = points_dict[key]
            # coords = [(0, 1), (1, 0), (1, 1), (0, 0)]
            center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
            new_dict[key] = sorted(coords, key=lambda coord: (-135 - np.rad2deg(np.arctan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
        return new_dict
    
    def create_json(result_dict):
        """Create a json file with the coordinates of the corners of the image."""
        with open(Calibration.path_to_calibration_output,"w") as f:
            json.dump(result_dict,f,indent=4)
            f.close()
        
    def create_sanity_check(result_dict):
        """Create a sanity check image of all founds coreners"""
        for key in result_dict:
            points = np.array(list(map(list, result_dict[key])))
            x_points = points[:,[0]]
            y_points = points[:,[1]]
            plt.scatter(x_points,y_points,marker=".",label=f"{key}")
        plt.legend(loc="best",fontsize="small")
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.savefig("test/sanity.jpg")
    
    def driver():
        """Driver function for the calibration process."""
        try:
            # Clearing the calibration file
            Calibration.clear_calibration_file(Calibration.path_to_intermediary)
            # Clearing the log file
            Calibration.clear_log_file(Calibration.path_to_calibration_log)
            # Iterating over all images
            Calibration.iterate(Calibration.path_to_calibration)
            # Creating the reference file
            result = Calibration.create_reference_file("truc")
            # Ordering the corners in clockwise order
            result_ordered = Calibration.order_corner_clockwise(result)
            # Creating the json file
            Calibration.create_json(result_ordered)
            # Creating a sanity check image
            Calibration.create_sanity_check(result_ordered)
            return 0
        except Exception as e:
            logging.exception("Exception occurred")
            return 1
        

if __name__ == "__main__":
    Calibration.driver()
