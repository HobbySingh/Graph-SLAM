import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


def draw_trajectory(traj, frame_id, x, y, z, draw_x, draw_y, true_x, true_y):
    
    cv2.circle(traj, (draw_x,draw_y), 1, (frame_id*255/4540,255-frame_id*255/4540,0), 1)
    cv2.circle(traj, (true_x,true_y), 1, (0,0,255), 2)
    cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
    text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(x,y,z)
    cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
    cv2.imshow('Trajectory', traj)
    

def convert_to_Rt(R,t):
    
    """
    converts to 3x4 transformation matrix
    """

    return np.hstack((R, t.reshape(-1,1)))

def rotation_to_quaternions(rotation: np.ndarray):

    """
    convert rotation matrix to quaternions vector
    """

    assert rotation.shape ==  (3,3)

    return R.from_matrix(rotation).as_quat()

def convert_to_4_by_4(Rt):

    try:
        assert Rt.shape==(3,4)
    except:
        print(Rt.shape)
        raise AssertionError("Input Matrix form should be of 3x4")

    return np.vstack((Rt, np.array([0,0,0,1])))


def getTransform(cur_pose, prev_pose):

    """
    Computes the error of the transformation between 2 poses
    """

    Rt = np.eye(4)
    # Rt[:3,:3] = cur_pose[:3,:3].T @ prev_pose[:3, :3]
    # Rt[:3, -1] = cur_pose[:3, :3].T @ (cur_pose[:3,-1] - prev_pose[:3, -1])
    Rt = np.linalg.inv(prev_pose)@cur_pose
    return Rt

def getError(cur_pose, prev_pose, cur_gt, prev_gt):

    """
    Computes the error of the transformation between 2 poses
    """
    
    error_t = np.linalg.norm((prev_pose[:3, -1] - cur_pose[:3,-1]) - (cur_gt[:3,-1] - prev_gt[:3,-1]))

    gt_prev_qt = rotation_to_quaternions(prev_gt[:3, :3])
    gt_qt = rotation_to_quaternions(cur_gt[:3, :3])
    gt_tform = gt_prev_qt * gt_qt.inverse()

    cur_qt = rotation_to_quaternions(prev_pose[:3, :3])
    prev_qt = rotation_to_quaternions(cur_pose[:3, :3])

    qt_tform = prev_qt * cur_qt.inverse()

    error_r = np.sum(np.rad2deg(quaternion.rotation_intrinsic_distance(gt_tform, qt_tform)))

    return error_r, error_t
    