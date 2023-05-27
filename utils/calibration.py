# Standard Python libraries
import pathlib

# Third-party libraries
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import cv2

# Local modules
from warp import Warp
from operations import CalibrationOperations as cop
from operations import FileOperations as fop
from operations import MathOperations as mop
from operations import LineBuilder

# Custom types
from custom_types import Path, Dataframe, DictPoints, NumpyArray

#! WARNING: CENTER CALIBRATION IS NOT WORKING YET

class ImageCalibration:
    """This class is used to calibrate the images."""

    def get_reference_points(
        path_to_image: Path, rotate: bool = True, resize: bool = True
    ) -> list[list[float]] | None:
        """This method is used to get the reference points from an image.

        Parameters
        ----------
        path_to_image : Path
            Path to the image.
        rotate : bool, optional
            Rotate the image before showing the image, by default True

        Returns
        -------
        list[list[float]] | None
            List of the reference points. None if the user did not select any points.
        """
        # Image loading
        name = path_to_image.stem
        img = fop.open_image_as_array(path_to_image)
        if rotate:
            img = np.flipud(img)
        if resize:
            img = cv2.resize(
                img, (0, 0), fx=0.5, fy=0.5
            )  # Lower the resolution to speed up the process
        # Plot generation
        fig, ax = plt.subplots()
        ax.set_title(name + " - Select the reference points.")
        ax.imshow(img)
        # Variable declaration
        var_data = []

        # Event handling
        def onclick(event):
            nonlocal var_data
            var_data.append([event.xdata, event.ydata])
            ax.plot(event.xdata, event.ydata, ".", c="b")
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("button_press_event", onclick)
        plt.show()
        # Output generation
        if len(var_data) != 4:
            # If the user did not select 4 points, the result is discarded.
            return None
        else:
            return cop.order_points_clockwise(var_data)

    def create_reference_dataframe(
        path_to_calibration_images: Path,
        calibration_positions: DictPoints,
        resize: bool = True,
    ) -> Dataframe:
        """This method is used to create the reference dataframe.

        Returns
        -------
        Dataframe
            Reference dataframe.

        Notes
        -----
        The reference dataframe contains the following columns:
        - name: name of the image
        - distance: distance of the image from the camera
        - 0, 1, 2, 3: x and y coordinates of the reference points
        """
        data = {
            path.stem: ImageCalibration.get_reference_points(path, resize=resize)
            for path in path_to_calibration_images.glob("*.jpg")
            if path.is_file()
        }
        data = {key: value for key, value in data.items() if value is not None}
        data = {key: data[key] for key in sorted(data.keys())}
        dataframe = pd.DataFrame.from_dict(data, orient="index")
        dataframe["name"] = sorted(dataframe.index)
        dataframe["distance"] = [
            calibration_positions[f"pic{name}"] for name in dataframe.index
        ]
        return dataframe


class CenterCalibration:
    """
    This class is used to define the center of the model for further analysis.
    """

    def prepare_image(
        path_to_image: Path,
        reference_dataframe: Dataframe,
        rotate: bool = True,
        resize: bool = False,
    ) -> NumpyArray:
        """This method is used to prepare the image for the center calibration.

        Parameters
        ----------
        path_to_image : Path
            Path to the image.
        reference_dataframe : Dataframe
            Reference dataframe for calibration data.
        rotate : bool, optional
            If set to true, the image is rotated, by default True
        resize : bool, optional
            If set to true, the image is resized, by default False

        Returns
        -------
        NumpyArray
            Prepared image.
        """
        img = Warp.warp(
            path_to_image, reference_dataframe, rotate=rotate, resize=resize
        )
        if img is None:
            return None
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.addWeighted(img, 3, np.zeros(img.shape, img.dtype), 0, 0)
            return img
        

    def launch(img: NumpyArray, title: str) -> dict[str, list[np.float32]]:
        """This method is used to launch the center calibration.

        Parameters
        ----------
        img : NumpyArray
            Prepared image.

        Returns
        -------
        dict[str, list[np.float32]]
            Dictionary containing the reference lines.
        """
        fig, ax = plt.subplots()
        ax.imshow(img)

        line = Line2D([500, 300], [500, 300], marker="o", markerfacecolor="blue")
        ax.add_line(line)
        linebuilder = LineBuilder(line)

        ax.set_title(f"click to create lines - {title}")
        plt.show()
        x_data = linebuilder.xs
        y_data = linebuilder.ys
        return np.array(
            [[x_data[0], y_data[0]], [x_data[1], y_data[1]]], dtype=np.float32
        )

    def get_reference_lines(img: NumpyArray) -> dict[str, list[np.float32]]:
        """This method is used to get the reference lines.

        Parameters
        ----------
        img : NumpyArray
            Prepared image.

        Returns
        -------
        dict[str, list[np.float32]]
            Dictionary containing the reference lines (slope, slope-intercept value).
        """
        wing_line = CenterCalibration.launch(img, title="wing line")
        body_line1 = CenterCalibration.launch(img, title="body line 1")
        body_line2 = CenterCalibration.launch(img, title="body line 2")
        if wing_line[0][0] == wing_line[1][1] or body_line1[0][0] == body_line1[1][1] or body_line2[0][0] == body_line2[1][1]:
            return None
        else:
            return {
                "wing_line": [
                    (wing_line[1][1] - wing_line[0][1])/ (wing_line[1][0] - wing_line[0][0]),
                    wing_line[0][1] - (wing_line[1][1] - wing_line[0][1])/ (wing_line[1][0] - wing_line[0][0]) * wing_line[0][0]
                ],
                "body_line1": [
                    (body_line1[1][1] - body_line1[0][1])/ (body_line1[1][0] - body_line1[0][0]),
                    body_line1[0][1] - (body_line1[1][1] - body_line1[0][1])/ (body_line1[1][0] - body_line1[0][0]) * body_line1[0][0]
                ],
                "body_line2": [
                    (body_line2[1][1] - body_line2[0][1])/ (body_line2[1][0] - body_line2[0][0]),
                    body_line2[0][1] - (body_line2[1][1] - body_line2[0][1])/ (body_line2[1][0] - body_line2[0][0]) * body_line2[0][0]
                ]
            }

    def intersection_points(
        points_dict: dict[str, list[np.float32]]
    ) -> list[np.float32]:
        """This method is used to get the intersection points of the reference lines.

        Parameters
        ----------
        points_dict : dict[str, list[np.float32]]
            Dictionary containing the reference lines.

        Returns
        -------
        list[np.float32]
            Coordinates of the center of the model.
        """
        wing_line = points_dict["wing_line"]
        body_line1 = points_dict["body_line1"]
        body_line2 = points_dict["body_line2"]
        try:
            intersection_point_body1 = (wing_line[1] - body_line1[1])/(body_line1[0] - wing_line[0])
            intersection_point_body2 = (wing_line[1] - body_line2[1])/(body_line2[0] - wing_line[0])
            return [
                (intersection_point_body1[0] + intersection_point_body2[0])/2,
                (intersection_point_body1[1] + intersection_point_body2[1])/2
            ]
        except:
            return None
        

    def create_reference_dataframe(
        path_to_folder: Path,
        calibration_dataframe: Dataframe,
        rotate: bool = True,
        resize: bool = False,
        print: bool = False,
    ) -> Dataframe:
        """This method is used to create the reference dataframe.

        Parameters
        ----------
        path_to_folder : Path
            Path to the calibration folder.
        calibration_dataframe : Dataframe
            Dataframe containing the calibration data.
        rotate : bool, optional
            If set to true, the images are rotated, by default True
        resize : bool, optional
            if set to true, the images are resized, by default False

        Returns
        -------
        Dataframe
            Dataframe containing the reference data.
        """
        result_dict = {}
        for path in path_to_folder.glob("*.jpg"):
            img = CenterCalibration.prepare_image(
                path, calibration_dataframe, rotate=rotate, resize=resize
            )
            if img is None:
                pass
            else:
                points_dict = CenterCalibration.get_reference_lines(img)
                intersection_points = CenterCalibration.intersection_points(points_dict)
                result_dict[f"{path_to_folder.stem}/{path.stem}"] = intersection_points
                if print:
                    fig, ax = plt.subplots()
                    ax.imshow(img)
                    ax.scatter(intersection_points[0], intersection_points[1], c="r")
                    ax.set_title(path.stem)
                    plt.savefig(f"debug/test-{path_to_folder.stem}-{path.stem}.png")
                    plt.clf()
        data = {key: value for key, value in result_dict.items() if value is not None}
        data = {key: data[key] for key in sorted(data.keys())}
        dataframe = pd.DataFrame.from_dict(data, orient="index")
        dataframe["name"] = sorted(dataframe.index)
        return dataframe
    
    def extract_center_from_dataframe(dataframe: Dataframe):
        list_center = []
        for row in dataframe.iterrows():
            row = row[1].to_dict()
            name = row["name"]
            if row["center"] != None:
                list_center.append({"label": f"{name}", "center": row["center"]})
        return list_center
    
    def create_center_dataframe(folder_to_pickle: Path):
        global_list_center = []
        for path in folder_to_pickle.glob("*.pkl"):
            name = path.stem.strip("result_")
            dataframe = fop.load_pickle_to_dataframe(path)
            list_center = CenterCalibration.extract_center_from_dataframe(dataframe)
            global_list_center += list_center
        result_dict = {}
        for center in global_list_center:
            label = center["label"].split("-")
            name = f"{label[0]}-{label[1]}-{label[2]}"
            if name not in result_dict.keys():
                result_dict[name] = []
        for center in global_list_center:
            label = center["label"].split("-")
            name = f"{label[0]}-{label[1]}-{label[2]}"
            result_dict[name].append(center["center"])
        center_dict = {}
        for key in result_dict.keys():
            center_array = np.array(result_dict[key],dtype=np.float32)
            center_dict[key] = [np.mean(center_array[:, 0]), np.mean(center_array[:, 1])]
        
        return center_dict

