# Standard Python libraries
import pathlib

# Third-party libraries
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2

# Local modules
from operations import CalibrationOperations as cop
from operations import FileOperations as fop
from operations import MathOperations as mop

# Custom types
from custom_types import Path, Dataframe, DictPoints, Coordinates, CalibrationVerification


class ImageCalibration:
    """
    This class is used to define the calibration points used to unwarp the raw images.
    """

    def __init__(
        self,
        path_to_calibration_images: Path,
        path_to_output_file: Path,
        calibration_positions: DictPoints,
    ) -> None:
        """Class constructor.

        Parameters
        ----------
        path_to_calibration_images : Path
            Path to the folder containing the calibration images.
        path_to_output_file : Path
            Path to where the output file will be saved.
        calibration_positions : DictPoints
            Positions of the calibration points on the test bench.
        """
        self.path_to_calibration_images = path_to_calibration_images
        self.path_to_output_file = path_to_output_file
        self.calibration_positions = calibration_positions

    def __repr__(self) -> str:
        """Class representation for debugging/logging purposes.

        Returns
        -------
        str
            Class representation
        """
        return f"ImageCalibration(path_to_calibration_images={self.path_to_calibration_images}, path_to_output_file={self.path_to_output_file} calibration_positions={self.calibration_positions})"

    def __str__(self) -> str:
        """Pretty print of the class representation for debugging purposes and interface.

        Returns
        -------
        str
            Pretty print of the class representation.
        """
        return self.__repr__

    def get_reference_points(
        path_to_image: Path, rotate: bool = True
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

    def create_reference_dataframe(self) -> Dataframe:
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
            path.stem: ImageCalibration.get_reference_points(path)
            for path in self.path_to_calibration_images.glob("*.jpg")
            if path.is_file()
        }
        data = {key: value for key, value in data.items() if value is not None}
        data = {key: data[key] for key in sorted(data.keys())}
        dataframe = pd.DataFrame.from_dict(data, orient="index")
        dataframe["name"] = sorted(dataframe.index)
        dataframe["distance"] = [
            self.calibration_positions[f"pic{name}"] for name in dataframe.index
        ]
        return dataframe


class ImageCalibrationVerification(ImageCalibration):
    """
    This class defines the methods used to verify the calibration of the camera.
    """
    def __init__(
        self,
        path_to_calibration_images: Path,
        path_to_output_file: Path,
        calibration_positions: DictPoints,
        points_of_interest: list[str],
    ) -> None:
        """Class constructor.

        Parameters
        ----------
        path_to_calibration_images : Path
            Path to the folder containing the calibration images.
        path_to_output_file : Path
            Path to where the output file will be saved.
        calibration_positions : DictPoints
            Positions of the calibration points on the test bench.
        points_of_interest : list[str]
            List of the points of interest that needs to be modified.
        """
        super().__init__(
            path_to_calibration_images, path_to_output_file, calibration_positions
        )
        self.points_of_interest = points_of_interest

    def __repr__(self) -> str:
        return f"ImageCalibrationVerification(path_to_calibration_images={self.path_to_calibration_images}, path_to_output_file={self.path_to_output_file} calibration_positions={self.calibration_positions}, points_of_interest={self.points_of_interest})"

    def __str__(self) -> str:
        return self.__repr__
    
    def check_reference_points(
        self,
    ) -> dict[str, dict[str, list[list[float]]]]:
        dataframe = fop.load_pickle_to_dataframe(self.path_to_output_file)
        regression_points = {f"p{i}": [] for i in range(4)}
        regression_points_x_distance = {f"p{i}": [] for i in range(4)}
        regression_points_y_distance = {f"p{i}": [] for i in range(4)}
        r2_regression_points = {f"p{i}": [] for i in range(4)}
        colors = {"0": "b", "1": "r", "2": "g", "3": "y"}
        for column in dataframe.columns:
            if column not in ["name", "distance"]:
                column_dict = dataframe[column].to_dict()
                column_num = dataframe.columns.get_loc(column)
                for key, value in column_dict.items():
                    regression_points[f"p{column_num}"].append(
                        np.array(value, dtype=np.float32)
                    )
                    regression_points_x_distance[f"p{column_num}"].append(
                        np.array(
                            [dataframe["distance"][key], value[0]], dtype=np.float32
                        )
                    )
                    regression_points_y_distance[f"p{column_num}"].append(
                        np.array(
                            [dataframe["distance"][key], value[1]], dtype=np.float32
                        )
                    )
                    plt.plot(value[0], value[1], ".", c=colors[str(column_num)])
                    plt.annotate(
                        key,
                        (value[0], value[1]),
                        color=colors[str(column_num)],
                        fontsize=6,
                        textcoords="offset points",
                        xytext=(5, 5),
                    )

        # Regression on the points
        def func(x, a, b):
            return a * x + b

        for i in range(4):
            xdata = np.array(
                [point[0] for point in regression_points[f"p{i}"]], dtype=np.float32
            )
            ydata = np.array(
                [point[1] for point in regression_points[f"p{i}"]], dtype=np.float32
            )
            popt, pcov = curve_fit(func, xdata, ydata)
            r2 = mop.compute_r2(xdata, ydata, func, popt)
            r2_regression_points[f"p{i}"].append(r2)
            plt.plot(xdata, func(xdata, *popt), c=colors[str(i)], linestyle="--")
        plt.title("Reference points")
        plt.xlabel("x [px]")
        plt.ylabel("y [px]")
        patches = [
            mpatches.Patch(
                color=colors[str(i)],
                label=f"p{i+1} - $R^2$ = "
                + str(np.round(r2_regression_points[f"p{i}"], 5)),
            )
            for i in range(4)
        ]
        plt.legend(handles=patches)
        plt.savefig
        plt.clf()
        return {
            "regression_points": regression_points,
            "regression_points_x_distance": regression_points_x_distance,
            "regression_points_y_distance": regression_points_y_distance,
            "r2_regression_points": r2_regression_points,
        }

    # def modify_reference_points(self) -> None:
    #     dataframe = fop.load_pickle_to_dataframe(self.path_to_output_file)
    #     for point in self.points_of_interest:
    #         new_result = ImageCalibration.get_reference_points(
    #             self.path_to_calibration_images / f"{point}.jpg"
    #         )
    #         if new_result is not None:
    #             dataframe.loc[point] = new_result
    #     fop.save_dataframe_to_pickle(dataframe, self.path_to_output_file)
    #     return None

    