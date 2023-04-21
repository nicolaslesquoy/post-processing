import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import tomli as tomllib
import numpy as np
from scipy.optimize import curve_fit

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

df = pd.read_pickle(path_to_intermediary)
# print(df.iloc[0])
# df.replace(None,"[None,None]",inplace=True)
for i in range(len(df.index)):
    row = df.iloc[i]
    row_array = row.values.tolist()[0:len(row)-1]
    for i in range(len(row_array)):
        if row_array[i] == None:
            row_array[i] = [None,None]
    row_array = np.array(row_array,dtype=np.float32)
    # print(row_array)
    # print(row_array[:,1])
    plt.scatter(row_array[:,0],row_array[:,1],marker="x")
plt.clf()

# Test interpolation
def interpolation(data: np.ndarray) -> np.ndarray:
    """Interpolate the data."""
    def func(x,a,b):
        # Basis function
        return a*x+b
    x = data[:,0]
    y = data[:,1]
    popt, pcov = curve_fit(func, x, y)
    return popt, pcov

# data = df.iloc[0].values.tolist()[0:len(df.iloc[0])-1]
# for i in range(len(data)):
#     if data[i] == None:
#         data[i] = [None,None]
# data = np.array(data,dtype=np.float32)
# popt, pcov = interpolation(data)



#     row = df.iloc[i]
#     print(row)
#     row_array = np.array(row.values.tolist()[1:len(row)-1], dtype=np.float32)
#     print(row_array)


# data = {"1": [[1,2],[3,4],[4,5],[6,7]],"3": [[16,17],[18,19],[None,None],[None,None]],"2": [[8,9],[10,11],[12,13],[14,15]]}

# # print(config)
# calibration_positions = config["measures_calibration"]
# print(calibration_positions)
# keys = list(data.keys())
# keys.sort()
# data = {key: data[key] for key in keys}
# df = pd.DataFrame.from_dict(data,orient="index")
# df.to_pickle("test/test.pkl")

# # df = pd.DataFrame({
# #     'p1': [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
# #     'p2': [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
# #     'p3': [[13.0, 14.0], [15.0, 16.0], [17.0, 18.0]],
# #     'p4': [[19.0, 20.0], [21.0, 22.0], [23.0, 24.0]],
# #     'distance': [1.0, 2.0, 3.0]
# # })

# # Select a row from the dataframe
# print(df)
# for i in range(len(df.index)):
#     row = df.iloc[i]
#     row_array = np.array(row.values.tolist()[0:len(row)], dtype=np.float32)
#     print(row_array)

# # Convert the row to a numpy array of arrays
# # row_array = np.array(row.values.tolist()[:-1], dtype=np.float32)

# test = [[16,17],[18,19],[None,None],[None,None]]
# # Operations.order_corners_clockwise(test)

# for i in range(len(test)):
#     plt.scatter(test[i][0],test[i][1])
# plt.clf()

# liste = [[1693.9655344655341, 2003.767732267732],[3363.4640359640343, 1999.4200799200798],[1798.3091908091903, 2964.598901098901],[3241.729770229769, 2947.2082917082917]]
# sorted_list = Operations.order_corners_clockwise(liste)
# print(sorted_list)

    


