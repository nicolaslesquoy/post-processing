import pandas as pd
import pathlib
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import tomli as tomllib
import numpy as np
import cv2
from PIL import Image

from operations import Operations

# df : row image number, p1, p2, p3, p4, distance

with open("test/config.toml","rb") as f:
    config = tomllib.load(f)

distance = config["measures_calibration"]

path_to_calibration = pathlib.Path(config["paths"]["path_to_calibration"])
path_to_camera_calibration = pathlib.Path(config["paths"]["path_to_camera_calibration"])
path_to_debug = pathlib.Path(config["paths"]["path_to_debug"])
    # path_to_logs = pathlib.Path(config["paths"]["path_to_logs"]) # TODO Use these files when implementing 'logging'
    # path_to_calibration_log = path_to_logs / "calibration.log"
path_to_intermediary = path_to_debug / "intermediary.pkl"
path_to_raw = pathlib.Path(config["paths"]["path_to_raw"])
path_to_images = path_to_raw / "incidence_std"

data = pd.read_pickle(path_to_intermediary)
data.to_string("test/intermediary.txt")

# Test interpolation
def compute_r2(xdata,ydata,func,popt):
    residuals = ydata - func(xdata,*popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ydata-np.mean(ydata))**2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

def interpolation(data: np.ndarray) -> np.ndarray:
    """Interpolate the data."""
    def func(x,a,b):
        # Basis function
        return a*x+b
    popt, pcov = curve_fit(func,data[:,0], data[:,1])
    return popt, compute_r2(data[:,0],data[:,1],func,popt), func(data[:,0],*popt)

# def create_random_row():
#     """Create a random row for the dataframe."""
#     row = []
#     result = {}
#     for i in range(6):
#         row.append([np.random.randint(0,1000),np.random.randint(0,1000)])
#     nm = ''.join(random.choices(string.ascii_lowercase, k=5))
#     dt = np.random.randint(0,1000)
#     return {"name": nm,"points": row,"distance":dt}

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

def iterate_over_rows(df):
    """Iterate over the rows of the dataframe."""
    for i in range(len(df.index)):
        row = df.iloc[i]
        row_array = row.values.tolist()[0:len(row)]
        points = np.array(row_array[1:len(row_array)-1],dtype=np.float32)
        plt.scatter(points[:,0],points[:,1],marker="x")
    plt.savefig("test_load_row.png")
    plt.clf()

def iterate_over_columns(df):
    """Iterate over the columns of the dataframe."""
    for i in range(1,df.shape[1]-1):
        column = df.iloc[:,i]
        column_list = column.values.tolist()[0:len(column)]
        column_array = np.array(column_list,dtype=np.float32)
        plt.scatter(column_array[:,0],column_array[:,1],marker="x")
        popt, r2, ydata = interpolation(column_array)
        plt.plot(column_array[:,0],ydata)
    plt.savefig("test_load_column.png")
    plt.clf()


# iterate_over_rows(data)
# iterate_over_columns(data)

def create_dst_points(Nx,Ny):
    """Create the destination points."""
    center = [Nx/2,Ny/2]
    dx,dy = 500,500
    p1 = [center[0] + dx, center[1] + dy]
    p2 = [center[0] - dx, center[1] + dy]
    p3 = [center[0] - dx, center[1] - dy]
    p4 = [center[0] + dx, center[1] - dy]
    dst_points = np.array(Operations.order_corners_clockwise([
        p1,p2,p3,p4
    ]),dtype=np.float32)
    return dst_points

def warp(img,img_points,dst_points):
    """Warp the image."""
    M = cv2.getPerspectiveTransform(img_points,dst_points)
    warped = cv2.warpPerspective(img,M,(img.shape[1],img.shape[0]))
    return warped

def read_file_name(path_to_file: pathlib.Path):
    """Read the file name from the path."""
    name = str(path_to_file.stem)
    incidence, angle, ref = name.split("-")
    return incidence, angle, ref

for path_to_file in path_to_images.glob("*.jpg"):
    img = np.asanyarray(Image.open(path_to_file))
    img = cv2.rotate(img,cv2.ROTATE_180)
    incidence, angle, reference = read_file_name(path_to_file)
    for i in range(len(data.index)):
        row = data.iloc[i]
        row_array = row.values.tolist()[0:len(row)]
        name = row_array[0]
        if name == reference:
            print(name, reference)
            points = np.array(row_array[1:len(row_array)-1],dtype=np.float32)
            dst_points = create_dst_points(img.shape[1],img.shape[0])
            warped = warp(img,points,dst_points)
            cv2.imwrite(str(path_to_debug / f"{incidence}-{angle}-{reference}.jpg"),warped)

# img = np.asanyarray(Image.open(path_to_calibration / "2.jpg"))
# img = cv2.rotate(img,cv2.ROTATE_180)
# for i in range(len(data.index)):
#     row = data.iloc[i]
#     row_array = row.values.tolist()[0:len(row)]
#     name = row_array[0]
#     img = np.asanyarray(Image.open(path_to_calibration / f"{name}.jpg"))
#     img = cv2.rotate(img,cv2.ROTATE_180)
#     points = np.array(row_array[1:len(row_array)-1],dtype=np.float32)
#     dst_points = create_dst_points(img.shape[1],img.shape[0])
#     warped = warp(img,points,dst_points)
#     cv2.imwrite(str(path_to_debug / f"{name}.png"),warped)

# iterate over images

