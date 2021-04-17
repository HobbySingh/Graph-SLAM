
import math
import numpy as np
from util import warp2pi

class PoseSE2():

    def __init__(self, position, orientation):
        self.position = position
        self.orientation = warp2pi(orientation)
        self.arr = np.array([position[0], position[1], warp2pi(orientation)], dtype=
        np.float64)

    