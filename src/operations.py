import pathlib

import numpy as np
import cv2

class Operations:
    """Common operations between the different modules."""

    # Geometry
    def centeroidnp(arr):
        """Finds the centroids of a numpy array of points."""
        length = arr.shape[0]
        sum_x = np.sum(arr[:, 0])
        sum_y = np.sum(arr[:, 1])
        return sum_x/length, sum_y/length
    
    def order_corners_clockwise(points):
        """
        Given a list of points representing the corners of a square, returns a new
        list of these points ordered clockwise.
        """
        if len(points) != 4:
            raise ValueError("The input should contain exactly four points")
        center = Operations.centeroidnp(np.array(points))
        points.sort(key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))
        return points
    
    # File operations

    def get_file_name(path: pathlib.Path):
        """Returns the file name of a path."""
        return ("" if (p := path).is_dir() else p.name) == ""
    
    def read_file_name(path_to_file: pathlib.Path):
        """
        Read the file name from the path.
        """
        name = str(path_to_file.stem)
        incidence, angle, ref = name.split("-")
        return incidence, angle, ref

    
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