import cv2
import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import tomli as tomllib
from PIL import Image

# from operations import Operations

with open("test/config.toml","rb") as f:
    config = tomllib.load(f)

path_to_calibration = pathlib.Path(config["paths"]["path_to_calibration"])
path = path_to_calibration / "3.jpg"
assert(path.is_file())
test_image = cv2.imread(str(path))

fig = plt.figure()
img = np.asarray(Image.open(path))
# img = cv2.rotate(img,cv2.ROTATE_180)
plt.imshow(img)
point = []
def onclick(event):
    global point; point.append([event.xdata, event.ydata])
    plt.plot(event.xdata, event.ydata, '.',c='b')
    fig.canvas.draw()
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
# 3    3  [1728.7467532467529, 1495.0924075924072]  [3298.2492507492498, 1490.7447552447552]   [3180.8626373626357, 2451.575924075924]  [1824.3951048951042, 2468.9665334665333]       9.9
dx = 1000
dy = 1000
cropped = test_image[int(float(point[0][0])):int(float(point[0][0]))+dy,int(float(point[0][1])):int(float(point[0][1]))+dx]
cv2.imwrite("test/cropped.jpg",cropped)