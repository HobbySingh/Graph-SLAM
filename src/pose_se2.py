import math
import numpy as np
from util import warp2pi


class PoseSE2:
    def __init__(self, position, orientation):
        self.position = position
        self.orientation = warp2pi(orientation)
        self.arr = np.array(
            [[position[0]], [position[1]], [warp2pi(orientation)]], dtype=np.float64
        )

    @staticmethod
    def from_array(arr):
        return PoseSE2((arr[0], arr[1]), warp2pi(arr[2]))

    @staticmethod
    def from_rt_matrix(mat):
        # mat: 3x3 2D [R,t]
        R = mat[:2, :2]
        x, y = mat[:2, 2]
        theta = np.arctan2(R[1, 0], R[0, 0])
        return PoseSE2.from_array([x, y, theta])

    def get_rt_matrix(self):
        x, y, yaw = self.arr
        return np.array(
            [[np.cos(yaw), -np.sin(yaw), x], [np.sin(yaw), np.cos(yaw), y], [0, 0, 1],],
            dtype="float64",
        )

    def __sub__(self, other):

        x = (self.position[0] - other.position[0]) * np.cos(other.orientation) + (
            self.position[1] - other.position[1]
        ) * np.sin(other.orientation)
        y = (other.position[0] - self.position[0]) * np.sin(other.orientation) + (
            self.position[1] - other.position[1]
        ) * np.cos(other.orientation)
        theta = warp2pi(self.orientation - other.orientation)

        return PoseSE2([x, y], theta)

    def __add__(self, other):

        x = (self.position[0] + other.position[0]) * np.cos(other.orientation) + (
            self.position[1] + other.position[1]
        ) * np.sin(other.orientation)
        y = (other.position[0] + self.position[0]) * np.sin(other.orientation) + (
            self.position[1] + other.position[1]
        ) * np.cos(other.orientation)
        theta = warp2pi(self.orientation + other.orientation)

        return PoseSE2([x, y], theta)

    def copy(self):
        return PoseSE2(self.position, self.orientation)
