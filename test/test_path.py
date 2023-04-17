from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


with open("config.toml","rb") as f:
    config = tomllib.load(f)
print(config)

send_click_to_log = config["parameters"]["send_onclick_to_log"]

path_to_calibration = Path(config["paths"]["path_to_calibration"])
path_to_camera_calibration = Path(config["paths"]["path_to_camera_calibration"])
path_to_debug = Path(config["paths"]["path_to_debug"])
path_to_intermediary = Path(config["paths"]["path_to_logs"]) / "intermediary.csv"