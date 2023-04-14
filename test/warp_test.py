import json
import cv2
import numpy as np
from functools import reduce
import operator

path_to_reference = r"/home/nlesquoy/Documents/Cours/EAEP-209/TP MAE 2022/Calibration/"
path_to_calibration_output = r"/home/nlesquoy/Documents/Cours/post-processing/src/calibration.json"
path_to_intermediary = r"/home/nlesquoy/Documents/Cours/post-processing/src/intermediary.csv"


objectpoints = [(0,0),(10,0),(10,10),(0,10)]
with open(path_to_calibration_output,"r") as f:
    data = json.load(f)
    f.close()
# print(data)
imagepoints = data["0"]
# new = []
# for element in imagepoints_raw:
#     # print(type(element))
#     e = tuple([np.float32(element[0]),np.float32(element[1])])
#     print(e)
#     new.append(e)
# print(imagepoints)
image = r"/home/nlesquoy/Documents/Cours/EAEP-209/TP MAE 2022/Calibration/0.jpg"

img = cv2.imread(image)

# Define the source points (the four corners of a rectangle in the original image)
src_points = np.float32(imagepoints)

# Define the destination points (the four corners of a smaller rectangle where we want to warp the image)
dst = [[1000, 1000], [2000, 1000], [2000, 2000], [1000, 2000]]

def create_dst(img,Nx,Ny):
    Nx,Ny = img.shape[1], img.shape[0]
    rect = [[0,0],[Ny,0],[Nx,Ny],[Nx,0]]
    x_m,y_m = Nx/2, Ny/2
    def l_to_r(x):
        return (rect[0][1] - rect[1][1])/(rect[0][0] - rect[1][0])*x
    def r_to_l(x):
        return (rect[3][1] - rect[2][1])/(rect[3][0] - rect[2][0])*x + Ny

def order_corner_clockwise(points):
    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), points), [len(points)] * 2))
    return sorted(points, key=lambda coord: (-135 - np.rad2deg(np.arctan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)

dst_points = np.float32(order_corner_clockwise(dst))
print(dst_points)

# Calculate the perspective transform matrix
M = cv2.getPerspectiveTransform(src_points, dst_points)
# [[1000, 1000], [1000, 2000], [2000, 2000], [2000, 1000]]
# Warp the image
print((img.shape[1], img.shape[0]))
warped_img = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))

cv2.imwrite("test/0_m.jpg",warped_img)