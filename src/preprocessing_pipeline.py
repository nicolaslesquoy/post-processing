# TODO : Add docstring
import pipeline
import numpy as np
import cv2


class PreprocessingPipeline(pipeline.Pipeline):
    def __init__(self, path_to_image: str, is_ref: bool):
        super().__init__(path_to_image, is_ref)
    
    def process(
        self,
        reference_points: np.ndarray,
        projection_points: np.ndarray,
        output_folder: str = "/out",
        rotate: bool = False,
    ) -> None:
        src = cv2.imread(self.path)
        if rotate:
            try:
                cv2.rotate(src, cv2.ROTATE_90_CLOCKWISE)
            except:
                raise Exception("Rotate failed")
            self.state = "rotated"
        # warp image
        img = cv2.imread(self.path)
        src_points = np.float32(reference_points)
        dst_points = np.float32(projection_points)
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        warped_img = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
        cv2.imwrite("".join(output_folder, "result"), warped_img)
        self.state = "unwarped"
        return None
