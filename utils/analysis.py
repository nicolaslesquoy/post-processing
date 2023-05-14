# Standard Python libraries

# Third party libraries
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import numpy as np
import pandas as pd

# Local packages
from operations import CalibrationOperations as cop
from operations import FileOperations as fop
from operations import ImageOperations as iop
from operations import MathOperations as mop
from warp import Warp

class Analysis:
    def draw_rectangle(name: str, img: np.ndarray):
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
            useblit=True,
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

    def get_center(name: str, img: np.ndarray):
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

    def driver(path_to_folder, calibration_dataframe):
        """Driver function for the analysis process."""
        result_dict = {}
        for path_to_file in path_to_folder.glob("*.jpg"):
            warped = Warp.warp(path_to_file, calibration_dataframe)
            result_int = {}
            # name = path_to_file.stem
            # ref = Operations.read_file_name(path_to_file)[2]
            try:
                name = path_to_file.stem + "-" + path_to_folder.stem
                result_int["name"] = name
                result_int["rectangle"] = Analysis.draw_rectangle(name, warped)
                result_int["center"] = Analysis.get_center(name, warped)
                result_dict[name] = result_int
            except:
                pass
        return result_dict
    
    def save_to_df(result_dict):
        keys = list(result_dict.keys())
        keys.sort()
        data = {key: result_dict[key] for key in keys}
        dataframe = pd.DataFrame.from_dict(data, orient="index")
        return dataframe