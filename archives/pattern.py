import cv2
import numpy as np
import matplotlib.pyplot as plt
  
image = cv2.imread('/home/nlesquoy/Documents/Cours/EAEP-209/processing/images/debug/vortex.png')
  
# Grayscale
image = cv2.rotate(image, cv2.ROTATE_180)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3, 3), 0)
enhanced = cv2.addWeighted(blur, 3, blur, 0, 0)
# TODO adjust threshold for each image to cancel the influence of the laser plane
ret, thresh = cv2.threshold(enhanced, 150, 255, cv2.THRESH_BINARY)
cv2.imwrite('/home/nlesquoy/Documents/Cours/EAEP-209/processing/images/debug/tresh1.png', thresh)

contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
                                      
# draw contours on the original image
image_copy = image.copy()
cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
cv2.imwrite('/home/nlesquoy/Documents/Cours/EAEP-209/processing/images/debug/contours.png', image_copy)

# find center of vortex with coutours points
img = cv2.imread('/home/nlesquoy/Documents/Cours/EAEP-209/processing/images/debug/tresh1.png')
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

# Set the input image
ss.setBaseImage(img)

# Run the selective search algorithm
ss.switchToSelectiveSearchFast()
rects = ss.process()

# Classify the regions by size
small_regions = []
medium_regions = []
large_regions = []
for rect in rects:
    x, y, w, h = rect
    area = w * h
    if area < 1000:
        small_regions.append(rect)
    elif area < 5000:
        medium_regions.append(rect)
    else:
        large_regions.append(rect)

# Draw rectangles around the classified regions and the interesting regions
for rect in small_regions:
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
for rect in medium_regions:
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
for rect in large_regions:
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

# Show the result
cv2.imwrite("debug.png", img) 