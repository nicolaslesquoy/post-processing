import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
import cv2

def to_gray(self, img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def import_as_np(path_to_image: str) -> np.ndarray:
    return np.asarray(Image.open(path_to_image))

path_to_image = r"images/Img3670.jpg"

# print(repr(import_as_np(path_to_image)))
p1 = (3126,1735)
p2 = (3058,987)
p3 = (2028,999)
p4 = (1941,1739)
patch1 = Circle(p1,radius=50)
patch2 = Circle(p2,radius=50)
patch3 = Circle(p3,radius=50)
patch4 = Circle(p4,radius=50)
plt.imshow(import_as_np(path_to_image))
plt.gca().add_patch(patch1)
plt.gca().add_patch(patch2)
plt.gca().add_patch(patch3)
plt.gca().add_patch(patch4)
plt.close()

# Load the image
img = cv2.imread(path_to_image)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("images/gray.jpg", gray)
cleaned = cv2.medianBlur(gray, 3)
cv2.imwrite("images/cleaned.jpg", cleaned)

# # Define the source points (the four corners of a rectangle in the original image)
# src_points = np.float32([list(p1), list(p2), list(p3), list(p4)])

# # Define the destination points (the four corners of a smaller rectangle where we want to warp the image)
# dst_points = np.float32([[100, 100], [400, 100], [400, 400], [100, 400]])

# # Calculate the perspective transform matrix
# M = cv2.getPerspectiveTransform(src_points, dst_points)

# # Warp the image
# warped_img = cv2.warpPerspective(cleaned, M, (img.shape[1], img.shape[0]))

# # Display the original and warped images
"""
# img = cv2.medianBlur(img,5)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

plt.subplot(121),plt.imshow(img)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
"""