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

    def driver(path_to_folder: Path, calibration_dataframe: Dataframe, resize: bool = True) -> Dataframe:
        """Driver function for the analysis process."""
        result_dict = {}
        list_index = calibration_dataframe.index.tolist()
        for path_to_file in path_to_folder.glob("*.jpg"):
            calibration_ref = cop.read_file_name(path_to_file)[2]
            if calibration_ref in list_index:
                warped = Warp.warp(path_to_file, calibration_dataframe, resize=resize)
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