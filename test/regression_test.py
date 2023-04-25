# Standard Python libraries
import pathlib
import json
import tomli as tomllib  # Make this import robust for Python 3.11

# Third party libraries
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import cv2

# Local libraries
import operations

class Calibration:
    with open("test/config.toml", "rb") as f:
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
    path_to_intermediary = path_to_debug / "intermediary.pkl"
    # Positions of the calibration points on the test bench
    calibration_positions = config["measures_calibration"]
        
    # Core functions

    def get_calibration_points(path_to_image: pathlib.Path,rotate: bool = True) -> None:
        """
        This function is used to calibrate the camera.
        It takes a path to an image as input and returns a dictionary with the coordinates of the corners of the image.
        """
        name = path_to_image.stem
        fig,ax = plt.subplots()
        img = np.asarray(Image.open(path_to_image))
        if rotate:
            img = cv2.rotate(img, cv2.ROTATE_180)
        ax.imshow(img)
        # Variable declaration
        var_data = []

        def onclick(event):
            nonlocal var_data; var_data.append([event.xdata,event.ydata])
            ax.plot(event.xdata, event.ydata, ".", c="b")
            fig.canvas.draw()
        
        cid = fig.canvas.mpl_connect("button_press_event", onclick)
        plt.show()
        if len(var_data) != 4:
            return None
        else:
            var_data = operations.Operations.order_corners_clockwise(var_data)
            distance = Calibration.calibration_positions["pic" + name]
            return {"name": name, "p1":var_data[0], "p2": var_data[1], "p3": var_data[2], "p4": var_data[3], "distance": distance}

# def create_random_row():
#     """Create a random row for the dataframe."""
#     row = []
#     result = {}
#     for i in range(6):
#         row.append([np.random.randint(0,1000),np.random.randint(0,1000)])
#     nm = ''.join(random.choices(string.ascii_lowercase, k=5))
#     dt = np.random.randint(0,1000)
#     return {"name": nm,"points": row,"distance":dt}

    def create_reference_dataframe(path_to_calibration_folder: pathlib.Path,distance: dict):
        """
        Iterate over the files in a folder and apply the calibration function to each of them.
        write the output to a pandas dataframe.
        """
        data = {}
        for path in path_to_calibration_folder.iterdir():
            if path.is_file():
                result = Calibration.get_calibration_points(path)
                if result != None:
                    name = result[list(result.keys())[0]]
                    data[name] = result
        keys = list(data.keys())
        keys.sort()
        data = {key: data[key] for key in keys}
        dataframe = pd.DataFrame.from_dict(data,orient="index")
        return dataframe

# def create_random_dataframe():
#     """Create a random dataframe."""
#     df_dict = {}
#     for i in range(10):
#         result_int = create_random_row()
#         name = result_int[list(result_int.keys())[0]]
#         df_dict[name] = result_int
#     print(len(df_dict))
#     keys = list(df_dict.keys())
#     keys.sort()
#     df_dict = {key: df_dict[key] for key in keys}
#     df = pd.DataFrame.from_dict(df_dict,orient="index")
#     return df

    
    def save_calibration_dataframe(dataframe: pd.DataFrame,path_to_intermediary: pathlib.Path):
        """Save the dataframe to a csv file."""
        dataframe.to_pickle(path_to_intermediary)
        return None
    
    def load_calibration_dataframe(path_to_intermediary: pathlib.Path):
        """Load the dataframe from a csv file."""
        # dtypes = {"0": int, 2: list[float], 3}
        dataframe = pd.read_pickle(path_to_intermediary)
        return dataframe

    # def create_sanity_check(data: pd.DataFrame) -> None:
    #     """Create a sanity check image of all founds coreners"""# for i in range(len(df.index)):
    #     for i in range(len(data.index)):
    #         row = data.iloc[i]
    #         # TODO Create function to convert the row to np array for plotting
    #         row_array = row.values.tolist()[0:len(row)-1]
    #         for j in range(len(row_array)):
    #             if row_array[j] == None:
    #                 row_array[j] = [None,None]
    #         row_array = np.array(row_array,dtype=np.float32)
    #         plt.scatter(row_array[:,0], row_array[:,1],marker="x",label=f"Image {i}")
    #     plt.legend(loc="best")    
    #     plt.savefig("test/sanity_check.png")
    
    def driver(iteration: bool = False):
        """Driver function for the calibration process."""
        if iteration:
            # Clearing the calibration file
            df = Calibration.create_reference_dataframe(Calibration.path_to_calibration, Calibration.calibration_positions)
            # Save configuration
            Calibration.save_calibration_dataframe(df, Calibration.path_to_intermediary)
            return df
        else:
            df = Calibration.load_calibration_dataframe(Calibration.path_to_intermediary)
            return df
        
    
if __name__ == "__main__":
    Calibration.driver(iteration=True)
    # Calibration.create_sanity_check(Calibration.load_calibration_dataframe_from_csv(Calibration.path_to_intermediary))


