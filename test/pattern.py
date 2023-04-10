import cv2
import numpy as np
import matplotlib.pyplot as plt
  
# Let's load a simple image with 3 black squares
image = cv2.imread('images/vortex.png')
  
# Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3, 3), 0)
enhanced = cv2.addWeighted(blur, 3, blur, 0, 0)
ret, thresh = cv2.threshold(enhanced, 100, 255, cv2.THRESH_BINARY)
cv2.imwrite('images/tresh1.jpg', thresh)

contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
                                      
# draw contours on the original image
image_copy = image.copy()
cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
cv2.imwrite('../images/contours.jpg', image_copy)

# find center of vortex with coutours points
img = cv2.imread("images/tresh1.jpg")
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

""" # Extract SIFT features from the large regions
sift = cv2.xfeatures2d.SIFT_create()
large_region_features = []
for rect in large_regions:
    x, y, w, h = rect
    region = img[y:y+h, x:x+w]
    gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray_region, None)
    large_region_features.append(descriptors)

# Define the reference features
reference_features = large_region_features[0]

# Compare the features of each large region to the reference features
matches = []
for i in range(len(large_regions)):
    similarity = cv2.compareHist(reference_features, large_region_features[i], cv2.HISTCMP_CORREL)
    matches.append((i, similarity))

# Sort the matches by similarity
matches.sort(key=lambda x: x[1], reverse=True)

# Select the most interesting regions
num_interesting_regions = 3
interesting_regions = []
for i in range(num_interesting_regions):
    index = matches[i][0]
    interesting_regions.append(large_regions[index])

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
for rect in interesting_regions:
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)

# Show the result
cv2.imwrite("images/classify.jpg", img) """