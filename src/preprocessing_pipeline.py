# Local library imports
import pipeline

# standard library imports
import os
import json
# third party imports
import numpy as np
import cv2

class ImageManipulation:
    """Manipulate image"""
    def rotate_image(image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

class Calibraton:
    """Create calibration.json"""
    pass

class Warp:
    # TODO ! switch path_to_calibration to toml file
    path_to_calibration = r"/home/nlesquoy/Documents/Projets/post-processing/src/calibration.json"
    # TODO : Add a method to calculate the src and dst points + complete the  + change config file to toml/config file
    def __init__(self, path_to_image, calibration: int = 1):
        reference = str(calibration)
        with open(Warp.path_to_calibration,"r") as f:
            reference_points = json.load(f)
            f.close()
        self.path_to_image = path_to_image
        self.src = reference_points[reference]["src"]
        self.dst = reference_points[reference]["dst"]
        self.calibration = calibration
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.Minv = cv2.getPerspectiveTransform(self.dst, self.src)

    def __str__(self) -> str:
        return f"Image {self.path_to_image} warped from {self.src} -> {self.dst} with calibration {self.calibration}"

    def warp(self):
        img = cv2.imread(self.path_to_image)
        return cv2.warpPerspective(img, self.M, self.img_size, flags=cv2.INTER_LINEAR)

    def unwarp(self):
        img = cv2.imread(self.path_to_image)
        return cv2.warpPerspective(img, self.Minv, self.img_size, flags=cv2.INTER_LINEAR)

class PreprocessingPipeline(pipeline.Pipeline):
    """Preprocessing pipeline for image processing"""

    def __init__(self, path_to_image: str, is_ref: bool):
        super().__init__(path_to_image, is_ref)
        self.state = "preprocessed"

    def __str__(self) -> str:
        return super().__str__(self)
    
    def __repr__(self) -> str:
        return f"PreprocessingPipeline({self.path}, {self.ref})"
    
    def log_file_info(self):
        """Extract information from image"""
        # TODO : Extract information from image based on file name 
        # = calibration, incidence and drift, image number + log information from ffmpeq
        pass
    
