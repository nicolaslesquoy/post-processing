# Standard Python libraries
import pathlib

# Third-party libraries
import matplotlib.pyplot as plt
import cv2

# Local packages
from operations import CalibrationOperations as cop
from operations import FileOperations as fop


class ImageCalibration:
    """
    This class is used to define the calibration points used to unwarp the raw images.
    """
    def __init__(
        self,
        path_to_calibration_images: pathlib.Path,
        path_to_output_file: pathlib.Path,
        calibration_positions: dict,
    ) -> None:
        self.path_to_calibration_images = path_to_calibration_images
        self.path_to_output_file = path_to_output_file
        self.calibration_positions = calibration_positions

    def __repr__(self) -> str:
        return f"ImageCalibration(path_to_calibration_images={self.path_to_calibration_images}, path_to_output_file={self.path_to_output_file} calibration_positions={self.calibration_positions})"
    
    def __str__(self) -> str:
        return self.__repr__
    
    def get_reference_points(path_to_image: pathlib.Path, rotate: bool = True) -> None:
        # Image loading
        name = path_to_image.stem
        img = fop.open_image_as_array(path_to_image)
        if rotate:
            img = cv2.rotate(img, cv2.ROTATE_180)
        # Plot generation
        fig, ax = plt.subplots()
        ax.set_title(name + " - Select the reference points.")
        ax.imshow(img, cmap="gray")
        fig.canvas.blit(fig.bbox)
        var_data = []

        def onclick(event):
            nonlocal var_data; var_data.append([event.xdata, event.ydata])
            ax.plot(event.xdata, event.ydata, ".", c="b")
            fig.canvas.draw_idle()
            fig.canvas.blit(fig.bbox)
            fig.canvas.flush_events()
    
        cid = fig.canvas.mpl_connect("button_press_event", onclick)
        plt.show()
        if len(var_data) == 0:
            return None
        else:
            return var_data
    
    def driver(path_to_folder: pathlib.Path):
        for path in path_to_folder.iterdir():
            if path.is_file():
                result = ImageCalibration.get_reference_points(path)

        
        


    

# # Core functions
#     def get_calibration_points(
#         path_to_image: pathlib.Path, rotate: bool = True
#     ) -> None:
#         """
#         This function is used to calibrate the camera.
#         It takes a path to an image as input and returns a dictionary with the coordinates of the corners of the image.
#         """
#         name = path_to_image.stem
#         fig, ax = plt.subplots()
#         img = np.asarray(Image.open(path_to_image))
#         if rotate:
#             img = cv2.rotate(img, cv2.ROTATE_180)
#         ax.imshow(img)
#         # Variable declaration
#         var_data = []

#         def onclick(event):
#             nonlocal var_data
#             var_data.append([event.xdata, event.ydata])
#             ax.plot(event.xdata, event.ydata, ".", c="b")
#             fig.canvas.draw()

#         cid = fig.canvas.mpl_connect("button_press_event", onclick)
#         plt.show()
#         if len(var_data) != 4:
#             return None
#         else:
#             var_data = Operations.order_corners_clockwise(var_data)
#             distance = Calibration.calibration_positions["pic" + name]
#             return {
#                 "name": name,
#                 "p1": var_data[0],
#                 "p2": var_data[1],
#                 "p3": var_data[2],
#                 "p4": var_data[3],
#                 "distance": distance,
#             }

#     def create_reference_dataframe(
#         path_to_calibration_folder: pathlib.Path, distance: dict
#     ):
#         """
#         Iterate over the files in a folder and apply the calibration function to each of them.
#         write the output to a pandas dataframe.
#         """
#         data = {}
#         for path in path_to_calibration_folder.iterdir():
#             if path.is_file():
#                 result = Calibration.get_calibration_points(path)
#                 if result != None:
#                     name = result[list(result.keys())[0]]
#                     data[name] = result
#         keys = list(data.keys())
#         keys.sort()
#         data = {key: data[key] for key in keys}
#         dataframe = pd.DataFrame.from_dict(data, orient="index")
#         return dataframe

#     def save_calibration_dataframe(
#         dataframe: pd.DataFrame, path_to_intermediary: pathlib.Path
#     ):
#         """Save the dataframe to a csv file."""
#         dataframe.to_pickle(path_to_intermediary)
#         return None

#     def load_calibration_dataframe(path_to_intermediary: pathlib.Path):
#         """Load the dataframe from a csv file."""
#         # dtypes = {"0": int, 2: list[float], 3}
#         dataframe = pd.read_pickle(path_to_intermediary)
#         return dataframe

#     def driver(
#         path_to_intermediary: pathlib.Path = path_to_intermediary,
#         path_to_calibration: pathlib.Path = path_to_calibration,
#         calibration_positions=calibration_positions,
#         iteration: bool = False,
#     ):
#         """Driver function for the calibration process."""
#         if iteration:
#             # Clearing the calibration file
#             df = Calibration.create_reference_dataframe(
#                 path_to_calibration, calibration_positions
#             )
#             # Save configuration
#             Calibration.save_calibration_dataframe(df, path_to_intermediary)
#             return df
#         else:
#             df = Calibration.load_calibration_dataframe(path_to_intermediary)
#             return df

class CameraCalibration:
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
    

