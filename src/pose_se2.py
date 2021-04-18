
import math
import numpy as np
from util import warp2pi

class PoseSE2():

    def __init__(self, position, orientation):
        self.position = position
        self.orientation = warp2pi(orientation)
        self.arr = np.array([[position[0]], [position[1]], [warp2pi(orientation)]], dtype=
        np.float64)

    def __sub__(self, other):

        x = (self.position[0] - other.position[0])*np.cos(other.orientation) + (self.position[1] - other.position[1])*np.sin(other.orientation)
        y = (other.position[0] - self.position[0])*np.sin(other.orientation) + (self.position[1] - other.position[1])*np.cos(other.orientation)
        theta = warp2pi(self.orientation - other.orientation)

        return PoseSE2([x, y], theta)