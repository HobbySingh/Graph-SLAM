import numpy as np
import scipy.io
import cv2
import copy
from .utils import *

class LoopClosure():
    def __init__(self, path, dataset, intrinsic_mat):
        gt_loop_data = scipy.io.loadmat(path + 'gnd_kitti07.mat')
        self.neighbours = gt_loop_data['gnd']            # This is a numpy array of shape (num_images, 1)
        self.dataset = dataset
        self.K = intrinsic_mat
        return

    def check_loop_closure(self, idx, frame_new):
        loop_closure_flag = False
        pose, matched_idx = None, None
        best_kp1, best_kp2, best_matches = None, None, None
        local_neighbours = self.neighbours[idx][0][0]         # numpy array of neighbours
        valid_neighbours = local_neighbours[local_neighbours < idx]

        # Check similarity with all valid neighbours and choose 1 or 2 to create edges
        max_num_matches = 0
        for img_idx in valid_neighbours:
            frame_old, _, _ = self.dataset[img_idx]
            # Check similarity using keypoint matches

            kp1, kp2, matches = self.find_matches(frame_new, frame_old)
            # Can also set min number of required matches

            if len(matches) > max_num_matches:
                max_num_matches = len(matches)
                target_frame = frame_old.copy()
                best_kp1 = kp1.copy()
                best_kp2 = kp2.copy()
                best_matches = matches.copy()
                matched_idx = img_idx
                # if(max_num_matches > 300):
                #     print("Length of best matches: ", max_num_matches)
                #     break

        # Compute R and t for maximally matching neighbours
        if max_num_matches >= 100:
            # import ipdb; ipdb.set_trace()
            matched_kp1 = []
            matched_kp2 = []
            self.DrawMatches(frame_new, target_frame, best_kp1, best_kp2, best_matches)
            # import ipdb; ipdb.set_trace()
            for mat in best_matches[:100]:
                matched_kp1.append(best_kp1[mat.queryIdx].pt)
                matched_kp2.append(best_kp2[mat.trainIdx].pt)

            matched_kp1 = np.array(matched_kp1)
            matched_kp2 = np.array(matched_kp2)
            E, _ = cv2.findEssentialMat(matched_kp1, matched_kp2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, mask = cv2.recoverPose(E, matched_kp1, matched_kp2, self.K)
            num_inliers = mask[mask > 0].shape[0]
            # num_outliers = mask.shape[0] - num_inliers
            inlier_ratio = num_inliers/mask.shape[0]
            # import ipdb; ipdb.set_trace()
            if inlier_ratio < 0.6:
                print("Found a matrix with very poor inlier ratio: ", inlier_ratio)
                return False, pose, matched_idx
            if (abs(t[1]) > abs(t[2]) or abs(t[0]) > abs(t[2])):
                print("Found unexpected translation ", t)
                return False, pose, matched_idx
            pose = convert_to_4_by_4(convert_to_Rt(R,t))
            print("New index: ", idx)
            print("Matched Index: ", matched_idx)
            print("Total Matches: ", len(best_matches))
            loop_closure_flag = True
            # cv2.imshow('Current', frame_new)
            # cv2.waitKey(0)
            # cv2.imshow('Target', target_frame)
            # cv2.waitKey(0)
        return loop_closure_flag, pose, matched_idx

    def DrawMatches(self, img1, img2, kp1, kp2, matches):
        # matches = sorted(matches, key = lambda x:x.distance)
        img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:],None, flags=2)
        cv2.namedWindow('Matches', cv2.WINDOW_NORMAL)
        cv2.imshow('Matches', img3)
        cv2.waitKey(0)
        return 

    def find_matches(self, img1, img2, return_ratio = 1):
        sift = cv2.SIFT_create()

        kp1, descriptors_1 = sift.detectAndCompute(img1,None)
        kp2, descriptors_2 = sift.detectAndCompute(img2,None)

        # Nearest matches for lowe's ratio test
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors_1,descriptors_2,k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.8*n.distance:
                good.append(m)
        matches = sorted(good, key = lambda x:x.distance)
        return kp1, kp2, matches
