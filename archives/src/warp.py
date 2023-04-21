import json
import tomli as tomllib
from pathlib import Path
import cv2

class Warp:
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)

    send_click_to_log = config["parameters"]["send_onclick_to_log"]

    path_to_calibration = Path(config["paths"]["path_to_calibration"])
    path_to_camera_calibration = Path(config["paths"]["path_to_camera_calibration"])
    path_to_debug = Path(config["paths"]["path_to_debug"])
    path_to_logs = Path(config["paths"]["path_to_logs"])
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