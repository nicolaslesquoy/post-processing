import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image

image = Image.open('../images/Img3805.jpg')
print(image)
# calculate mean value from RGB channels and flatten to 1D array
vals = np.mean(image,axis=2).flatten()
# plot histogram with 255 bins
b, bins, patches = plt.hist(vals, 255)
plt.xlim([0,255])
plt.savefig("histogram.png")