import numpy as np
from matplotlib import pyplot as plt
import argparse
import cv2
def parse_argument():
    
    parser=argparse.ArgumentParser()
    parser.add_argument('--path', required=True)
    return parser.parse_args()

def compute_trans_err(gt, pose):
    t1 = gt[:3,3]
    t2 = pose[:3,3]
    err = np.linalg.norm(t1-t2)
    return err

def compute_rot_err(gt, pose):
    mat = gt[:3,:3]@pose[:3,:3].T
    r, _ = cv2.Rodrigues(mat)
    err = np.linalg.norm(r)
    return err

if  __name__ == '__main__':

    sequence = 'data/' + '00'
    optimized_poses = np.load(sequence+'Poses.npy')
    gt_poses = np.load(sequence+'Gt.npy')
    raw_poses = np.load(sequence+'Raw.npy')

    # import ipdb; ipdb.set_trace()

    t_err_raw = [compute_trans_err(gt_poses[i], raw_poses[i]) for i in range(gt_poses.shape[0])]
    t_err_optimized = [compute_trans_err(gt_poses[i], optimized_poses[i]) for i in range(gt_poses.shape[0])]

    rot_err_raw = [compute_rot_err(gt_poses[i], raw_poses[i]) for i in range(gt_poses.shape[0])]
    rot_err_optimized = [compute_rot_err(gt_poses[i], optimized_poses[i]) for i in range(gt_poses.shape[0])]

    # figure, axis = plt.subplots(1, 2)
    # plt.set_title("Errors")

    t = np.arange(len(t_err_raw))
    plt.plot(t, t_err_raw, 'r', label='raw')
    plt.plot(t, t_err_optimized, 'b', label='optimized')
    plt.legend()
    # plt.set_title("Translation Error")
    plt.title("Translation Error")
    plt.xlabel('Frame')
    plt.ylabel('Error')
    plt.grid()

    # axis[0].plot(t, t_err_raw, 'r', label='raw')
    # axis[0].plot(t, t_err_optimized, 'b', label='optimized')
    # axis[0].legend()
    # axis[0].set_title("Translation Error")
    # axis[0].set_xlabel('Frame')
    # axis[0].set_ylabel('Error')
    # axis[0].grid()

    # axis[1].plot(t, rot_err_raw, 'r', label='raw')
    # axis[1].plot(t, rot_err_optimized, 'b', label='optimized')
    # axis[1].legend()
    # axis[1].set_title("Rotational Error")
    # axis[1].set_xlabel('Frame')
    # axis[1].set_ylabel('Error')
    # axis[1].grid()

    # Combine all the operations and display
    plt.show() 