import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

class PostProcessing:
    # unwarping images using cv2

    def __init__(self,path_to_image: str,is_ref: bool):
        self.path = path_to_image
        self.state = "raw"
        self.ref = is_ref
    
    def __str__(self) -> str:
        return "{0}: [state = {1}] [ref = {2}]".format(self.path,self.state,self.ref)
    
    def _import(self) -> np.ndarray:
        return np.asarray(Image.open(self.path))
    
    def _warp_and_clean(self,reference_points: np.ndarray, projection_points: np.ndarray, output_folder: str = "/out", rotate: bool = False) -> None:
        """Unwarp and clean the image (rotate, gray scale, median blur and warping)"""
        img = cv2.imread(self.path)
        if rotate:
            try:
                cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
            except:
                raise Exception("Rotate failed")
            self.state = "rotated"
        else:
            self.state = "not rotated"
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.state = gray
        cleaned = cv2.medianBlur(gray, 5)
        self.state = cleaned
        src_points = np.float32(reference_points)
        dst_points = np.float32(projection_points)
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        warped_img = cv2.warpPerspective(cleaned, M, (cleaned.shape[1], cleaned.shape[0]))
        cv2.imwrite("".join(output_folder,"result"), warped_img)
        self.state = "unwarped"
        return None
