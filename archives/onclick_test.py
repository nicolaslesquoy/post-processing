import os
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Circle
from PIL import Image
import numpy as np
import pandas as pd
from functools import reduce
import operator
# import cv2

class Calibration:
    # TODO Move path to config
    path_to_reference = r"/home/nlesquoy/Documents/Cours/EAEP-209/TP MAE 2022/Calibration/"
    path_to_calibration_output = r"/home/nlesquoy/Documents/Cours/post-processing/src/calibration.json"
    path_to_intermediary = r"/home/nlesquoy/Documents/Cours/post-processing/src/intermediary.csv"

    def calibration(path_to_image):
        fig = plt.figure()
        img = np.asarray(Image.open(path_to_image))
        name = os.path.basename(path_to_image) 
        plt.imshow(img)
        def onclick(event):
            with open(Calibration.path_to_intermediary,"a") as f:
                f.write(f"{name},{event.xdata},{event.ydata}" + "\n")
                f.close()
            print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %(event.button, event.x, event.y, event.xdata, event.ydata))
            plt.plot(event.xdata, event.ydata, '.',c='b')
            fig.canvas.draw()
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        return 0
    
    def iterate(path_to_folder):
        for filename in os.listdir(path_to_folder):
            f = os.path.join(path_to_folder, filename)
            # checking if it is a file
            if os.path.isfile(f):
                Calibration.calibration(f)
        return None
    
    def create_reference_file(path_to_calibration_file):
        df = pd.read_csv(Calibration.path_to_intermediary,sep=",",header=None,index_col=0)
        result_dict = {}
        for image,data in df.groupby(0):
            points = []
            for index,row in data.iterrows():
                points.append((int(float(row[1])),int(float(row[2]))))
            result_dict[image.strip('.jpg')] = points
        return result_dict

    def order_corner_clockwise(points_dict):
        new_dict = {}
        for key in points_dict:
            coords = points_dict[key]
            # coords = [(0, 1), (1, 0), (1, 1), (0, 0)]
            center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
            new_dict[key] = sorted(coords, key=lambda coord: (-135 - np.rad2deg(np.arctan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
        return new_dict
    
    def create_json(result_dict):
        with open(Calibration.path_to_calibration_output,"w") as f:
            json.dump(result_dict,f,indent=4)
            f.close()
        
    def create_sanity_check(result_dict):
        colors = cm.rainbow(np.linspace(0, 0.01, len(result_dict)))
        for key in result_dict:
            points = np.array(list(map(list, result_dict[key])))
            # print(points)
            x_points = points[:,[0]]
            # print(x_points)
            y_points = points[:,[1]]
            # print(y_points)
            plt.scatter(x_points,y_points,marker=".",label=f"{key}")
        plt.legend(loc="best",fontsize="small")
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.savefig("test/sanity.jpg")
        

result = Calibration.create_reference_file("truc")
result_ordered = Calibration.order_corner_clockwise(result)
Calibration.create_json(result_ordered)
# # Sanity check
Calibration.create_sanity_check(result_ordered)
# for key in result_ordered:
#     for i in range(len(result_ordered[key])):
#         if int(key) % 2 == 0:
#             plt.plot(result_ordered[key][i][0],result_ordered[key][i][1],".",c="b")
#         else:
#             plt.plot(result_ordered[key][i][0],result_ordered[key][i][1],".",c="r")
#         plt.annotate(str(f"({i}/{key})"),(result_ordered[key][i][0],result_ordered[key][i][1]))
# plt.show()
