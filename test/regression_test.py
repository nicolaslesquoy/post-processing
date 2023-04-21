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
        if len(var_data) != 4 and len(var_data) == 2:
            if name == "1" or name == "0":
                if var_data[0][0] > var_data[1][0]:
                    var_data = [[None,None],[None,None],var_data[0],var_data[1]]
                else:
                    var_data = [[None,None],[None,None],var_data[1],var_data[0]]
            elif name == "8":
                if var_data[0][0] > var_data[1][0]:
                    [var_data[0],var_data[1],[None,None],[None,None]]
                else:
                    [var_data[1],var_data[0],[None,None],[None,None]]
            else:
                raise ValueError(f"The image {name} does not contain exactly four points.")
        else:
            var_data = operations.Operations.order_corners_clockwise(var_data) 
        return {name: var_data}

    def create_reference_dataframe(path_to_calibration_folder: pathlib.Path,distance: dict):
        """
        Iterate over the files in a folder and apply the calibration function to each of them.
        write the output to a pandas dataframe.
        """
        data = {}
        for path in path_to_calibration_folder.iterdir():
            if path.is_file():
                result = Calibration.get_calibration_points(path)
                key = list(result.keys())[0]
                data[key] = result[key]
        keys = list(data.keys())
        keys.sort()
        data = {key: data[key] for key in keys}
        dataframe = pd.DataFrame.from_dict(data,orient="index")
        distance = sorted(distance.values())
        dataframe["distance"] = distance
        return dataframe
    
    def save_calibration_dataframe(dataframe: pd.DataFrame,path_to_intermediary: pathlib.Path):
        """Save the dataframe to a csv file."""
        dataframe.to_pickle(path_to_intermediary)
        return None
    
    def load_calibration_dataframe(path_to_intermediary: pathlib.Path):
        """Load the dataframe from a csv file."""
        # dtypes = {"0": int, 2: list[float], 3}
        dataframe = pd.read_pickle(path_to_intermediary)
        return dataframe

    def create_sanity_check(data: pd.DataFrame) -> None:
        """Create a sanity check image of all founds coreners"""# for i in range(len(df.index)):
        for i in range(len(data.index)):
            row = data.iloc[i]
            # TODO Create function to convert the row to np array for plotting
            row_array = row.values.tolist()[0:len(row)-1]
            for j in range(len(row_array)):
                if row_array[j] == None:
                    row_array[j] = [None,None]
            row_array = np.array(row_array,dtype=np.float32)
            plt.scatter(row_array[:,0], row_array[:,1],marker="x",label=f"Image {i}")
        plt.legend(loc="best")    
        plt.savefig("test/sanity_check.png")
    
    def driver(iteration: bool = False):
        """Driver function for the calibration process."""
        if iteration:
            # Clearing the calibration file
            df = Calibration.create_reference_dataframe(Calibration.path_to_calibration, Calibration.calibration_positions)
            # Save configuration
            Calibration.save_calibration_dataframe(df, Calibration.path_to_intermediary)
            Calibration.create_sanity_check(df)
            return df
        else:
            df = Calibration.load_calibration_dataframe(Calibration.path_to_intermediary)
            Calibration.create_sanity_check(df)
            return df

class Interpolation:
    # Interpolation functions
    def interpolation(data: np.ndarray) -> np.ndarray:
        """Interpolate the data."""
        def func(x,a,b):
            # Basis function
            return a*x+b
        x = data[:,0]
        y = data[:,1]
        popt, pcov = curve_fit(func, x, y)
        return popt, pcov

    def create_interpolation_dataframe(data: pd.DataFrame) -> pd.DataFrame:
        """Create a dataframe with the interpolation data."""
        # 1. Pixel interpolation for position in image
        # 2. Pixel interpolation for distance
        
    
if __name__ == "__main__":
    Calibration.driver(iteration=False)
    # Calibration.create_sanity_check(Calibration.load_calibration_dataframe_from_csv(Calibration.path_to_intermediary))


""" # img = np.asarray(Image.open(path_to_calibration / "3.jpg"))
# img = cv2.rotate(img, cv2.ROTATE_180)
# plt.imshow(img)
# plt.show()

with open(path_to_calibration_result, "r") as f:
    calibration_result = json.load(f)

for key, value in calibration_result.items():
    value = np.array(value)
    # print(value[:,0])
    # print(value[:,1])
    plt.scatter(value[:,0], value[:,1],marker="x")
    for i in range(len(value)):
        plt.text(value[i,0], value[i,1], key)
plt.savefig("test_image.jpg")
plt.clf()

# print(calibration_result)
liste_p1,liste_p2,liste_p3,liste_p4 = [],[],[],[]
for key, value in calibration_result.items():
    liste_p1.append(value[0])
    liste_p2.append(value[1])
    liste_p3.append(value[2])
    liste_p4.append(value[3])
arr_p1 = np.array(liste_p1)
arr_p2 = np.array(liste_p2)
arr_p3 = np.array(liste_p3)
arr_p4 = np.array(liste_p4)

# for path in path_to_calibration.iterdir():
#     if path.is_file():
#         name = path.stem
#         fig,ax = plt.subplots()
#         img = np.asarray(Image.open(path))
#         img = cv2.rotate(img, cv2.ROTATE_180)
#         ax.imshow(img)
#         ax.set_title(name)
#         plt.show()

points_0 = [[None,None],[1639,2790],[3477,2790],[None,None]]
points_1 = [[None,None],[1649,2363],[3407,2363],[None,None]]
points_8 = [[1916,873],[None,None],[None,None],[3011,860]]
liste_p1.append(points_0[0])
liste_p1.append(points_1[0])
liste_p2.append(points_0[1])
liste_p2.append(points_1[1])
liste_p3.append(points_8[0])
liste_p3.append(points_8[1])

def func(x, a, b):
    return a * x + b

popt_p1, pcov_p1 = curve_fit(func, arr_p1[:,0], arr_p1[:,1])
popt_p2, pcov_p2 = curve_fit(func, arr_p2[:,0], arr_p2[:,1])
popt_p3, pcov_p3 = curve_fit(func, arr_p3[:,0], arr_p3[:,1])
popt_p4, pcov_p4 = curve_fit(func, arr_p4[:,0], arr_p4[:,1])

# print(popt_p1,pcov_p1)

plt.plot(arr_p1[:,0], arr_p1[:,1], 'o', label='data_1')
plt.plot(arr_p2[:,0], arr_p2[:,1], 'o', label='data_2')
plt.plot(arr_p3[:,0], arr_p3[:,1], 'o', label='data_3')
plt.plot(arr_p4[:,0], arr_p4[:,1], 'o', label='data_4')
plt.plot(arr_p1[:,0], func(arr_p1[:,0], *popt_p1), 'r-', label='fit_1')
plt.plot(arr_p2[:,0], func(arr_p2[:,0], *popt_p2), 'r-', label='fit_2')
plt.plot(arr_p3[:,0], func(arr_p3[:,0], *popt_p3), 'r-', label='fit_3')
plt.plot(arr_p4[:,0], func(arr_p4[:,0], *popt_p4), 'r-', label='fit_4')
plt.legend()
plt.savefig("test_reg.jpg")
plt.clf()

calibration_measures = {
    "0": 0,
    "1": 2.7,
    "2": 5.5,
    "3": 9.9,
    "4": 14.1,
    "5": 17.7,
    "6": 20.8,
    "7": 23.8,
    "8": 28.3
}

liste_p1_dist_x,liste_p1_dist_y = [],[] # en bas à gauche
liste_p2_dist_x,liste_p2_dist_y = [],[] # en haut à gauche
liste_p3_dist_x,liste_p3_dist_y = [],[] # en haut à droite
liste_p4_dist_x,liste_p4_dist_y = [],[] # en bas à droite

calibration_result["0"] = points_0
calibration_result["1"] = points_1
calibration_result["8"] = points_8

for key, value in calibration_measures.items():
    calibration_result[key].append(value)

print(calibration_result) 

for key, value in calibration_result.items():
    if key not in ["0","1","8"]:
        liste_p1_dist_x.append([value[len(value) - 1],value[0][0]])
        liste_p1_dist_y.append([value[len(value) - 1],value[0][1]])
        liste_p2_dist_x.append([value[len(value) - 1],value[1][0]])
        liste_p2_dist_y.append([value[len(value) - 1],value[1][1]])
        liste_p3_dist_x.append([value[len(value) - 1],value[2][0]])
        liste_p3_dist_y.append([value[len(value) - 1],value[2][1]])
        liste_p4_dist_x.append([value[len(value) - 1],value[3][0]])
        liste_p4_dist_y.append([value[len(value) - 1],value[3][1]])
    elif key == "0" or key == "1":
        liste_p2_dist_x.append([value[len(value) - 1],value[1][0]])
        liste_p2_dist_y.append([value[len(value) - 1],value[1][1]])
        liste_p3_dist_x.append([value[len(value) - 1],value[2][0]])
        liste_p3_dist_y.append([value[len(value) - 1],value[2][1]])
    elif key == "8":
        liste_p1_dist_x.append([value[len(value) - 1],value[0][0]])
        liste_p1_dist_y.append([value[len(value) - 1],value[0][1]])
        liste_p4_dist_x.append([value[len(value) - 1],value[3][0]])
        liste_p4_dist_y.append([value[len(value) - 1],value[3][1]])
    else:
        raise Exception("Error")

arr_p1_dist_x = np.array(liste_p1_dist_x)
arr_p1_dist_y = np.array(liste_p1_dist_y)
arr_p2_dist_x = np.array(liste_p2_dist_x)
arr_p2_dist_y = np.array(liste_p2_dist_y)
arr_p3_dist_x = np.array(liste_p3_dist_x)
arr_p3_dist_y = np.array(liste_p3_dist_y)
arr_p4_dist_x = np.array(liste_p4_dist_x)
arr_p4_dist_y = np.array(liste_p4_dist_y)

# print(arr_p2_dist_y)
# print(arr_p3_dist_y)

popt_p1_dist_x, pcov_p1_dist_x = curve_fit(func, arr_p1_dist_x[:,0], arr_p1_dist_x[:,1])
popt_p1_dist_y, pcov_p1_dist_y = curve_fit(func, arr_p1_dist_y[:,0], arr_p1_dist_y[:,1])
popt_p2_dist_x, pcov_p2_dist_x = curve_fit(func, arr_p2_dist_x[:,0], arr_p2_dist_x[:,1])
popt_p2_dist_y, pcov_p2_dist_y = curve_fit(func, arr_p2_dist_y[:,0], arr_p2_dist_y[:,1])
popt_p3_dist_x, pcov_p3_dist_x = curve_fit(func, arr_p3_dist_x[:,0], arr_p3_dist_x[:,1])
popt_p3_dist_y, pcov_p3_dist_y = curve_fit(func, arr_p3_dist_y[:,0], arr_p3_dist_y[:,1])
popt_p4_dist_x, pcov_p4_dist_x = curve_fit(func, arr_p4_dist_x[:,0], arr_p4_dist_x[:,1])
popt_p4_dist_y, pcov_p4_dist_y = curve_fit(func, arr_p4_dist_y[:,0], arr_p4_dist_y[:,1])

#plt.plot(arr_p1_dist_x[:,0], arr_p1_dist_x[:,1], 'o', label='data_x1')
#plt.plot(arr_p1_dist_y[:,0], arr_p1_dist_y[:,1], 'o', label='data_y1')
plt.plot(arr_p2_dist_x[:,0], arr_p2_dist_x[:,1], 'o', label='data_x2')
plt.plot(arr_p2_dist_y[:,0], arr_p2_dist_y[:,1], 'o', label='data_y2')
plt.plot(arr_p3_dist_x[:,0], arr_p3_dist_x[:,1], 'o', label='data_x3')
plt.plot(arr_p3_dist_y[:,0], arr_p3_dist_y[:,1], 'o', label='data_y3')
#plt.plot(arr_p4_dist_x[:,0], arr_p4_dist_x[:,1], 'o', label='data_x4')
#plt.plot(arr_p4_dist_y[:,0], arr_p4_dist_y[:,1], 'o', label='data_y4')
#plt.plot(arr_p1_dist_x[:,0], func(arr_p1_dist_x[:,0], *popt_p1_dist_x), 'r-', label='fit_x1')
#plt.plot(arr_p1_dist_y[:,0], func(arr_p1_dist_y[:,0], *popt_p1_dist_y), 'r-', label='fit_y1')
plt.plot(arr_p2_dist_x[:,0], func(arr_p2_dist_x[:,0], *popt_p2_dist_x), 'r-', label='fit_x2')
plt.plot(arr_p2_dist_y[:,0], func(arr_p2_dist_y[:,0], *popt_p2_dist_y), 'r-', label='fit_y2')
plt.plot(arr_p3_dist_x[:,0], func(arr_p3_dist_x[:,0], *popt_p3_dist_x), 'r-', label='fit_x3')
plt.plot(arr_p3_dist_y[:,0], func(arr_p3_dist_y[:,0], *popt_p3_dist_y), 'r-', label='fit_y3')
#plt.plot(arr_p4_dist_x[:,0], func(arr_p4_dist_x[:,0], *popt_p4_dist_x), 'r-', label='fit_x4')
#plt.plot(arr_p4_dist_y[:,0], func(arr_p4_dist_y[:,0], *popt_p4_dist_y), 'r-', label='fit_y4')
plt.legend(loc="best")
plt.savefig("test_reg_dist.jpg")
plt.clf()
 """

# liste_distance = [float(value) for key, value in calibration_measures.items()]
# print(liste_distance)

# Nouvelle version

    # Cleaning functions

"""     def clear_intermediary_file(path_to_intermediary_file: pathlib.Path) -> bool:
        if path_to_intermediary_file.exists():
            path_to_intermediary_file.unlink()
            return True
        else:
            return False """
