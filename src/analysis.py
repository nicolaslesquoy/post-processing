import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import numpy as np
import cv2
from operations import FileOperations as fop
from operations import CalibrationOperations as cop
import calibration
from warp import Warp

# Custom types
from custom_types import Path, Dataframe, NumpyArray

from custom_types import NumpyArray

class Analysis:
    def draw_rectangle(name: str, img: NumpyArray, type: str = "stable"):
        fig, ax = plt.subplots()
        ax.imshow(img)
        var_data = []
        if type == "stable":
            color = "green"
            title = name + " - Select the rectangle (stable)"
        elif type == "unstable":
            color = "red"
            title = name + " - Select the rectangle (unstable)"
        elif type == "fuselage":
            color = "blue"
            title = name + " - Select the rectangle (fuselage)"
        else:
            color = "white"
            title = name + " - Select the rectangle (unknown)"

        def select_callback(eclick, erelease):
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            rect = plt.Rectangle(
                (min(x1, x2), min(y1, y2)),
                np.abs(x1 - x2),
                np.abs(y1 - y2),
                facecolor="none",
                edgecolor=color,
                linewidth=2,
            )
            ax.add_patch(rect)
            nonlocal var_data
            var_data.append({"a": [x1, y1], "b": [x2, y2]})

        plt.title(title)
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
    
    def get_center(folder_name: str, file_name: str, center_dataframe: Dataframe):
        """Get the center of the model."""
        name = f"{folder_name}/{file_name}"
        center = center_dataframe.iloc[name].to_list()
        return center

    def draw_center(name: str, img: NumpyArray):
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


    def driver(path_to_folder: Path, calibration_dataframe: Dataframe, center_dataframe: Dataframe, resize: bool = True, draw_center: bool = True) -> Dataframe:
        """Driver function for the analysis process."""
        result_dict = {}
        list_index = calibration_dataframe.index.tolist()
        for path_to_file in path_to_folder.glob("*.jpg"):
            calibration_ref = cop.read_file_name(path_to_file)[2]
            if calibration_ref in list_index:
                img = Warp.warp(path_to_file, calibration_dataframe, resize=resize)
                # th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
                warped = cv2.addWeighted(img, 3, np.zeros(img.shape, img.dtype), 0, 0)
                # warped = cv2.addWeighted(add_weighted, 0.5, th, 0.5, 0)
                result_int = {}
                try:
                    name = path_to_file.stem
                    result_int["name"] = name
                    result_int["rectangle_stable"] = Analysis.draw_rectangle(name, warped, type="stable")
                    result_int["rectangle_unstable"] = Analysis.draw_rectangle(name, warped, type="unstable")
                    result_int["rectangle_fuselage"] = Analysis.draw_rectangle(name, warped, type="fuselage")
                    if draw_center:
                        result_int["center"] = Analysis.draw_center(name, warped)
                    else:
                        result_int["center"] = Analysis.get_center(path_to_folder.stem, path_to_file.stem, center_dataframe)
                except:
                    pass
                result_dict[name] = result_int
        return pd.DataFrame.from_dict(result_dict, orient="index")