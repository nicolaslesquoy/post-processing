import pandas as pd

from calibration import ImageCalibration
from custom_types import Path, Dataframe, DictPoints, Coordinates, CalibrationVerification

class CalibrationVerification:

    def modify_dataframe(dataframe: Dataframe, points_of_interest: list[str], path_to_folder: Path) -> Dataframe:
        """This method is used to modify the dataframe with the calibration points.

        Parameters
        ----------
        dataframe : Dataframe
            Dataframe containing the calibration points.

        Returns
        -------
        Dataframe
            Dataframe with the calibration points.
        """
        # Create a copy of the dataframe
        dataframe = dataframe.copy()
        # Add the calibration points to the dataframe
        for point in points_of_interest:
            for index, rows in dataframe.iterrows():
                if rows["name"] == point:
                    path_to_image = path_to_folder / f"{point}.jpg"
                    result = ImageCalibration.get_reference_points(path_to_image)
                    
        # Return the modified dataframe
        return dataframe