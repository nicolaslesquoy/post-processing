# Standard Python libraries
import pathlib
from typing import NewType

# Third-party libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# Local packages
from operations import CalibrationOperations as cop
from operations import FileOperations as fop
from custom_types import Path, Dataframe, DictPoints



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
        self.path_to_calibration_images = path_to_calibration_images
        self.path_to_output_file = path_to_output_file
        self.calibration_positions = calibration_positions

    def __repr__(self) -> str:
        return f"ImageCalibration(path_to_calibration_images={self.path_to_calibration_images}, path_to_output_file={self.path_to_output_file} calibration_positions={self.calibration_positions})"

    def __str__(self) -> str:
        return self.__repr__

    def get_reference_points(path_to_image: Path, rotate: bool = True) -> list[list[float]] | None:
        # Image loading
        name = path_to_image.stem
        img = fop.open_image_as_array(path_to_image)
        if rotate:
            img = np.flipud(img)
            # img = cv2.rotate(img, cv2.ROTATE_180)
        img = cv2.resize(
            img, (0, 0), fx=0.5, fy=0.5
        )  # Lower the resolution to speed up the process
        # Plot generation
        fig, ax = plt.subplots()
        ax.set_title(name + " - Select the reference points.")
        ax.set_xticks([])
        ax.set_yticks([])
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
        if len(var_data) == 0:
            return None
        else:
            return var_data

    def create_reference_dataframe(self) -> Dataframe:

        data = {
            path.stem: ImageCalibration.get_reference_points(path)
            for path in self.path_to_calibration_images.glob("*.jpg")
            if path.is_file()
        }
        data = {key: value for key, value in data.items() if value is not None}
        data = {key: data[key] for key in sorted(data.keys())}
        dataframe = pd.DataFrame.from_dict(data, orient="index")
        dataframe["name"] = sorted(dataframe.index)
        dataframe["distance"] = [self.calibration_positions[f"pic{name}"] for name in dataframe.index]
        return dataframe


class ImageCalibrationVerification(ImageCalibration):
    def __init__(
        self,
        path_to_calibration_images: Path,
        path_to_output_file: Path,
        calibration_positions: DictPoints,
        points_of_interest: list[str],
    ) -> None:
        super().__init__(
            path_to_calibration_images, path_to_output_file, calibration_positions
        )
        self.points_of_interest = points_of_interest
    
    def __repr__(self) -> str:
        return f"ImageCalibrationVerification(path_to_calibration_images={self.path_to_calibration_images}, path_to_output_file={self.path_to_output_file} calibration_positions={self.calibration_positions}, points_of_interest={self.points_of_interest})"
    
    def __str__(self) -> str:
        return self.__repr__
    
    def modify_reference_points(self) -> None:
        dataframe = fop.load_pickle_to_dataframe(self.path_to_output_file)
        for point in self.points_of_interest:
            new_result = ImageCalibration.get_reference_points(self.path_to_calibration_images / f"{point}.jpg")
            if new_result is not None:
                dataframe.loc[point] = new_result
        fop.save_dataframe_to_pickle(dataframe, self.path_to_output_file)
        return None
    
    def check_reference_points(self):
        dataframe = fop.load_pickle_to_dataframe(self.path_to_output_file)
        for column in dataframe.columns:
            if column not in ["name", "distance"]:
                column_dict = dataframe[column].to_dict()
                print(column_dict)


class CameraCalibration:
    #! TODO Implement this class
    def __init__(
        self,
        path_to_calibration_images: pathlib.Path,
        path_to_output_file: pathlib.Path,
        grid_parameters: tuple,
    ) -> None:
        self.path_to_calibration_images = path_to_calibration_images
        self.path_to_output_file = path_to_output_file
        self.grid_parameters = grid_parameters

    def __repr__(self) -> str:
        return f"CameraCalibration(path_to_calibration_images={self.path_to_calibration_images}, path_to_output_file={self.path_to_output_file} grid_parameters={self.grid_parameters})"

    def __str__(self) -> str:
        return self.__repr__
