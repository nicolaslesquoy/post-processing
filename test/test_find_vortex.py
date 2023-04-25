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

with open("config.toml","rb") as f:
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

# Iterate over images
# for path_to_file in path_to_images.glob("*.jpg"):
#     img = np.asanyarray(Image.open(path_to_file))
#     img = cv2.rotate(img,cv2.ROTATE_180)
#     incidence, angle, reference = read_file_name(path_to_file)
#     for i in range(len(data.index)):
#         row = data.iloc[i]
#         row_array = row.values.tolist()[0:len(row)]
#         name = row_array[0]
#         if name == reference:
#             print(name, reference)
#             points = np.array(row_array[1:len(row_array)-1],dtype=np.float32)
#             dst_points = create_dst_points(img.shape[1],img.shape[0])
#             warped = warp(img,points,dst_points)
#             cv2.imwrite(str(path_to_debug / f"{incidence}-{angle}-{reference}.jpg"),warped)

# test_image = path_to_debug / "25-0-7.jpg"

# name = path_to_image.stem
#         fig,ax = plt.subplots()
#         img = np.asarray(Image.open(path_to_image))
#         if rotate:
#             img = cv2.rotate(img, cv2.ROTATE_180)
#         ax.imshow(img)
#         # Variable declaration
#         var_data = []

#         def onclick(event):
#             nonlocal var_data; var_data.append([event.xdata,event.ydata])
#             ax.plot(event.xdata, event.ydata, ".", c="b")
#             fig.canvas.draw()
        
#         cid = fig.canvas.mpl_connect("button_press_event", onclick)
#         plt.show()
#         if len(var_data) != 4:
#             return None
#         else:
#             var_data = operations.Operations.order_corners_clockwise(var_data)
#             distance = Calibration.calibration_positions["pic" + name]
#             return {"name": name, "p1":var_data[0], "p2": var_data[1], "p3": var_data[2], "p4": var_data[3], "distance": distance}

test_image = path_to_images / "25-0-4.jpg"
img = np.asanyarray(Image.open(test_image))
img = cv2.rotate(img,cv2.ROTATE_180)
data = pd.read_pickle(path_to_intermediary)
incidence, angle, reference = read_file_name(test_image)

for i in range(len(data.index)):
    row = data.iloc[i]
    row_array = row.values.tolist()[0:len(row)]
    name = row_array[0]
    if name == reference:
        points = np.array(row_array[1:len(row_array)-1],dtype=np.float32)
        dst_points = create_dst_points(img.shape[1],img.shape[0])
        warped = warp(img,points,dst_points)

fig, ax = plt.subplots()
line = ax.imshow(warped)

def line_select_callback(eclick, erelease):
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata

    rect = plt.Rectangle((min(x1,x2),min(y1,y2)), np.abs(x1-x2), np.abs(y1-y2), facecolor='none', edgecolor='red', linewidth=2)
    ax.add_patch(rect)
    print("({:.3f}, {:.3f}) --> ({:.3f}, {:.3f})".format(x1, y1, x2, y2))

def onclick(event):
    print([event.xdata,event.ydata])
    ax.plot(event.xdata, event.ydata, ".", c="b")
    fig.canvas.draw()

rs = RectangleSelector(ax, line_select_callback, useblit=False, button=[1],minspanx=5, minspany=5, spancoords='pixels',interactive=True)
plt.show()
plt.savefig("test/test_vortex.png")

# img = np.asanyarray(Image.open(test_image))
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.GaussianBlur(img,(5,5),0)
# enhanced = cv2.addWeighted(img, 3, img, 0, 0)

# ret, thresh = cv2.threshold(enhanced, 150, 255, cv2.THRESH_BINARY)
# cv2.imwrite('/home/nlesquoy/Documents/Cours/EAEP-209/processing/images/debug/test-tresh1.png', thresh)

# contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
                                      
# # draw contours on the original image
# image_copy = img.copy()
# cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
# cv2.imwrite('/home/nlesquoy/Documents/Cours/EAEP-209/processing/images/debug/test-contours.png', image_copy)

# # find center of vortex with coutours points
# img = cv2.imread('/home/nlesquoy/Documents/Cours/EAEP-209/processing/images/debug/test-tresh1.png')
# ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

# # Set the input image
# ss.setBaseImage(img)

# # Run the selective search algorithm
# ss.switchToSelectiveSearchFast()
# rects = ss.process()

# Classify the regions by size
# small_regions = []
# medium_regions = []
# large_regions = []
# for rect in rects:
#     x, y, w, h = rect
#     area = w * h
#     if area < 1000:
#         small_regions.append(rect)
#     elif area < 5000:
#         medium_regions.append(rect)
#     else:
#         large_regions.append(rect)

# # Draw rectangles around the classified regions and the interesting regions
# for rect in small_regions:
#     x, y, w, h = rect
#     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
# for rect in medium_regions:
#     x, y, w, h = rect
#     cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
# for rect in large_regions:
#     x, y, w, h = rect
#     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

# # Show the result
# cv2.imwrite("debug.png", img) 