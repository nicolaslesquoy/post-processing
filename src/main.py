# Standard Python libraries
import pathlib
import json
import tomli as tomllib  # Make this import robust for Python 3.11

# Third party libraries
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

from PIL import Image
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import cv2

# Local libraries
from operations import Operations

with open("config.toml", "rb") as f:
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
path_to_raw = pathlib.Path(config["paths"]["path_to_raw"])
path_to_images_incidence_std = path_to_raw / "incidence_std"
path_to_images_derapage_std = path_to_raw / "derapage_std"
path_to_intermediary = path_to_debug / "intermediary.pkl"
# Positions of the calibration points on the test bench
calibration_positions = config["measures_calibration"]

class Calibration:  
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
            var_data = Operations.order_corners_clockwise(var_data)
            distance = Calibration.calibration_positions["pic" + name]
            return {"name": name, "p1":var_data[0], "p2": var_data[1], "p3": var_data[2], "p4": var_data[3], "distance": distance}

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

    def save_calibration_dataframe(dataframe: pd.DataFrame,path_to_intermediary: pathlib.Path):
        """Save the dataframe to a csv file."""
        dataframe.to_pickle(path_to_intermediary)
        return None
    
    def load_calibration_dataframe(path_to_intermediary: pathlib.Path):
        """Load the dataframe from a csv file."""
        # dtypes = {"0": int, 2: list[float], 3}
        dataframe = pd.read_pickle(path_to_intermediary)
        return dataframe
    
    def driver(path_to_intermediary: pathlib.Path = path_to_intermediary, path_to_calibration: pathlib.Path = path_to_calibration, calibration_positions = calibration_positions, iteration: bool = False):
        """Driver function for the calibration process."""
        if iteration:
            # Clearing the calibration file
            df = Calibration.create_reference_dataframe(path_to_calibration, calibration_positions)
            # Save configuration
            Calibration.save_calibration_dataframe(df, path_to_intermediary)
            return df
        else:
            df = Calibration.load_calibration_dataframe(path_to_intermediary)
            return df

class Analysis:
    def create_dst_points(Nx,Ny):
        """
        Create the destination points.
        """
        center = [Nx/2,Ny/2]
        dx,dy = 500,500
        p1 = [center[0] + dx, center[1] + dy]
        p2 = [center[0] - dx, center[1] + dy]
        p3 = [center[0] - dx, center[1] - dy]
        p4 = [center[0] + dx, center[1] - dy]
        dst_points = np.array(Operations.order_corners_clockwise([p1,p2,p3,p4]),dtype=np.float32)
        return dst_points

    def get_image(path_to_image):
        """Get the image from the path."""
        img = np.asarray(Image.open(path_to_image))
        return img

    def get_reference_points(path_to_image,dataframe: pd.DataFrame):
        """Get the reference points from the dataframe."""
        try:
            ref_image = Operations.read_file_name(path_to_image)[2]
            reference = dataframe[dataframe["name"] == ref_image].to_dict(orient="records")[0]
            return np.array([reference["p1"],reference["p2"],reference["p3"],reference["p4"]],dtype=np.float32)
        except:
            return None
    
    def dewarp(path_to_image, dataframe: pd.DataFrame):
        """Dewarp the image."""
        img = Analysis.get_image(path_to_image)
        img = cv2.rotate(img,cv2.ROTATE_180)
        dst_points = Analysis.create_dst_points(img.shape[1],img.shape[0])
        reference_points = Analysis.get_reference_points(path_to_image,dataframe)
        try:
            M = cv2.getPerspectiveTransform(reference_points,dst_points)
            warped = cv2.warpPerspective(img,M,(img.shape[1],img.shape[0]))
            return warped
        except:
            return None
    
    def draw_rectangle(name: str,img: np.ndarray):
        fig, ax = plt.subplots()
        print(type(img))
        ax.imshow(img)
        var_data = []
        def select_callback(eclick, erelease):
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            rect = plt.Rectangle((min(x1,x2),min(y1,y2)), np.abs(x1-x2), np.abs(y1-y2), facecolor='none', edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            nonlocal var_data; var_data.append({"a": [x1,y1],"b": [x2,y2]})
            # print("({:.3f}, {:.3f}) --> ({:.3f}, {:.3f})".format(x1, y1, x2, y2))
        plt.title(name)
        rs = RectangleSelector(ax, select_callback, useblit=False, button=[1],minspanx=5, minspany=5, spancoords='pixels',interactive=True)
        plt.show()
        if len(var_data) == 0:
            return None
        else:
            return var_data
    
    def get_center(name: str,img: np.ndarray):
        fig,ax = plt.subplots()
        ax.imshow(img)
        # Variable declaration
        var_data = []

        def onclick(event):
            nonlocal var_data; var_data.append([event.xdata,event.ydata])
            ax.plot(event.xdata, event.ydata, ".", c="b")
            fig.canvas.draw()
        plt.title(name)
        cid = fig.canvas.mpl_connect("button_press_event", onclick)
        plt.show()
        if len(var_data) == 0:
            return None
        else:
            return var_data[0]

    def driver(path_to_folder,calibration_dataframe):
        """Driver function for the analysis process."""
        result_dict = {}
        for path_to_file in path_to_folder.glob("*.jpg"):
            warped = Analysis.dewarp(path_to_file,calibration_dataframe)
            result_int = {}
            print(path_to_file)
            name = path_to_file.stem
            # ref = Operations.read_file_name(path_to_file)[2]
            try:
                result_int["name"] = name
                result_int["rectangle"] = Analysis.draw_rectangle(name,warped)
                result_int["center"] = Analysis.get_center(name,warped)
                result_dict[name] = result_int
            except:
                pass
        return result_dict
    
    def save_to_df(result_dict):
        keys = list(result_dict.keys())
        keys.sort()
        data = {key: result_dict[key] for key in keys}
        dataframe = pd.DataFrame.from_dict(data,orient="index")
        return dataframe
    
    def save_to_pickle(dataframe,path_to_pickle):
        dataframe.to_pickle(path_to_pickle)

    def load_to_df(path_to_pickle):
        return pd.read_pickle(path_to_pickle)

    def get_vortex_position(center: list, rectangle: dict):
        if rectangle != None and center != None:
            x1,y1 = rectangle["a"]
            x2,y2 = rectangle["b"]
            center_vortex_x, center_vortex_y = [(x1 + x2)/2,(y1 + y2)/2]
            dist_x = center[0] - center_vortex_x
            dist_y = center[1] - center_vortex_y
            return [dist_x,dist_y]
        else:
            return None

    def get_conversion(Nx,Ny):
        reference = Analysis.create_dst_points(Nx,Ny)
        # print(reference)
        delta_nx = abs(reference[1][0] - reference[0][0])
        delta_ny = abs(reference[2][1] - reference[0][1])
        delta_x = 10 # cm
        delta_y = 10 # cm
        dx = delta_x/delta_nx
        dy = delta_y/delta_ny
        return [dx,dy]

    def prepare_points(result_dataframe: pd.DataFrame, conversion: list):
        dx, dy = conversion
        result_dict = {"15": [], "20": [], "25": []}
        for index, row in result_dataframe.iterrows():
            result_x, result_y = [],[]
            name = row["name"]
            informations = name.split("-")
            distance = calibration_positions["pic" + informations[2]]
            center = row["center"]
            rectangles = row["rectangle"]
            if center != None and rectangles != None:
                for i in range(len(rectangles)):
                    vortex_position = Analysis.get_vortex_position(center,rectangles[i])
                    x,y = abs(vortex_position[0]),abs(vortex_position[1])
                    x,y = x*dx,y*dy
                    result_x.append([distance,x])
                    result_y.append([distance,y])
                point = {"label": f"{informations[0]}/{informations[1]}/{informations[2]}","x": result_x,"y": result_y}
                result_dict[str(informations[0])].append(point)
        return result_dict

    def plot_points(result_dict):
        color = {"15": "r", "20": "b", "25": "g"}
        for key in result_dict:
            for i in range(len(result_dict[key])):
                label = result_dict[key][i]["label"].split("/")
                for j in range(len(result_dict[key][i]["x"])):
                    plt.scatter(result_dict[key][i]["x"][j][0],result_dict[key][i]["x"][j][1],color=color[key],marker="x")
                    plt.annotate(f"$b$={label[1]}°/plan n°{label[2]}",(result_dict[key][i]["x"][j][0],result_dict[key][i]["x"][j][1]),textcoords="offset points",xytext=(0,-10),fontsize=6)
            plt.title(f"Evolution de la position du tourbillon à $i$ = {key}° (x)")
            plt.xlabel("Distance vortex - caméra (cm)")
            plt.ylabel("Distance vortex - centre (cm)")
            plt.savefig(f"position_vortex_x_{key}.png")
            plt.clf()

        # for key in result_dict:   
        #     for i in range(len(result_dict[key])):
        #         for j in range(len(result_dict[key][i]["y"])):
        #             plt.scatter(result_dict[key][i]["y"][j][0],result_dict[key][i]["y"][j][1],color=color[key],marker="x")
            # # result_x, result_y = np.array(result_x),np.array(result_y)
            # plt.title(f"Evolution de la position du tourbillon à $i$ = {key}° (x)")
            # # plt.scatter(result_x[:,0],result_x[:,1],marker="x")
            # plt.xlabel("Distance vortex - caméra (cm)")
            # plt.ylabel("Distance vortex - centre (cm)")

    
"""
{'15': [{'label': '15/0/3', 'x': [[9.9, 2.8539834359189125]], 'y': [[9.9, 0.3490370274886482]]},
{'label': '15/0/4', 'x': [[14.1, 4.60851148851149]], 'y': [[14.1, 0.586933066933068]]}, 
{'label': '15/0/5', 'x': [[17.7, 5.841461764042402], [17.7, 0.04960716702652917]], 'y': [[17.7, 1.003828687441594], [17.7, 0.06908343269634087]]},
{'label': '15/0/6', 'x': [[20.8, 6.884898327478959], [20.8, 0.006130643550009154]], 'y': [[20.8, 1.565677161548133], [20.8, 0.17442841029938336]]}, 
{'label': '15/0/7', 'x': [[23.8, 7.766830588766066]], 'y': [[23.8, 1.248939705455839]]}], 
'20': [{'label': '20/0/3', 'x': [[9.9, 2.4533582546485695]], 'y': [[9.9, 0.3118316522187524]]}, 
{'label': '20/0/4', 'x': [[14.1, 4.083727885018197]], 'y': [[14.1, 0.736979407689089]]}, 
{'label': '20/0/5', 'x': [[17.7, 5.173706293706291]], 'y': [[17.7, 1.195604395604396]]}, 
{'label': '20/0/6', 'x': [[20.8, 6.347572427572422]], 'y': [[20.8, 1.6303696303696324]]}], 
'25': [{'label': '25/0/3', 'x': [[9.9, 2.1396861203312665]], 'y': [[9.9, 0.28089742515549687]]}, 
{'label': '25/0/4', 'x': [[14.1, 3.5433366633366585]], 'y': [[14.1, 0.9130069930069954]]}, 
{'label': '25/0/5', 'x': [[17.7, 4.723473945409423]], 'y': [[17.7, 1.4695281492700907]]}]}

"""

if __name__ == "__main__":
    calibration_dataframe = Calibration.driver(iteration=False)
    # result_dict = {}
    # result_int = {}
    # test_path = path_to_images_incidence_std / "15-0-7.jpg"    result_dict = Analysis.driver(path_to_images_incidence_std,calibration_dataframe)
    # df = Analysis.save_to_df(result_dict)
    # path_to_pickle_incidence_std = path_to_images_incidence_std / "analysis_incidence.pkl"
    # Analysis.save_to_pickle(df,path_to_pickle_incidence_std)
    # name = test_path.stem
    # ref = Operations.read_file_name(test_path)[2]
    # result_int["name"] = name
    # # print(Analysis.get_reference_points(test_path,calibration_dataframe))
    # warped = Analysis.dewarp(test_path,calibration_dataframe)
    # rectangles = None
    # result_int["rectangles"] = rectangles
    # center = None
    # result_int["center"] = center
    # result_dict[name] = result_int
    # df = Analysis.save_to_df(result_dict)
    # print(df)
    # result_dict = Analysis.driver(path_to_images_incidence_std,calibration_dataframe)
    # df = Analysis.save_to_df(result_dict)
    Nx = 4928
    Ny = 3264
    conversion = Analysis.get_conversion(Nx,Ny)
    # print(conversion)
    path_to_pickle_incidence_std = path_to_images_incidence_std / "analysis_incidence.pkl"
    # Analysis.save_to_pickle(df,path_to_pickle_incidence_std)
    df = Analysis.load_to_df(path_to_pickle_incidence_std)
    result_dict = Analysis.prepare_points(df,conversion)
    Analysis.plot_points(result_dict)