from typing import NewType
import pathlib
import pandas as pd
import numpy as np

# Type definitions
Path = NewType("Path", pathlib.Path)
Dataframe = NewType("Dataframe", pd.DataFrame)
NumpyArray = NewType("NumpyArray", np.ndarray)
DictPoints = NewType("DictPoints", dict[str, list[list[float]]])
Coordinates = NewType("Coords", list[list[float]])