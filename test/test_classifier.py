import cv2

# Load the input image
input_image = cv2.imread("input_image.jpg")

# Create a selective search object
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

# Set the input image for selective search
ss.setBaseImage(input_image)

# Run selective search to generate region proposals
ss.switchToSelectiveSearchFast()
rects = ss.process()

# Create a cascade classifier object
cascade_classifier = cv2.CascadeClassifier("cascade_classifier.xml")

# Select proposed images using the cascade classifier
proposed_images = []
for rect in rects:
    # Convert the region proposal to grayscale
    x, y, w, h = rect
    roi = input_image[y:y+h, x:x+w]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Detect objects in the grayscale region proposal using the cascade classifier
    objects = cascade_classifier.detectMultiScale(gray_roi, scaleFactor=1.1, minNeighbors=5)

    # Add the region proposal to the list of proposed images if an object is detected
    if len(objects) > 0:
        proposed_images.append(rect)

# Draw rectangles around the proposed images
for rect in proposed_images:
    x, y, w, h = rect
    cv2.rectangle(input_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the input image with proposed images
cv2.imshow("Input Image", input_image)
cv2.waitKey(0)
