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
    
