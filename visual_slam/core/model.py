import cv2
import numpy as np

from .optimizer import PoseGraph
from .geocom.features import featureTracking

from .utils import *
import pdb
from .loop_closure import LoopClosure

KMIN_NUM_FEATURE = 1500
optimize = False

class VisualSLAM():
   
    def __init__(self, camera_intrinsics, ground_pose, dataset, args):
        
        self.K = camera_intrinsics
        self.ground_pose = ground_pose
        self.args = args
        self.feature_tracker = featureTracking
        self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
        self.cur_R = None
        self.cur_t = None
        self.curRaw_R = None
        self.curRaw_t = None
        self.prev_frame = None
        self.trueX, self.trueY, self.trueZ = 0, 0, 0
        self.poses = []
        self.raw = []
        self.gt = []
        # np_gt = np.load('Gt.npy')
        # np_poses = np.load('Poses.npy')
        # self.gt = [np_gt[i] for i in range(np_gt.shape[0])]
        # self.poses = [np_poses[i] for i in range(np_poses.shape[0])]
        self.errors = []
        self.loop_closure_count = 0
        self.pose_graph = PoseGraph(verbose = True)
        self.loop_closure = LoopClosure(args.gt_loops, dataset, self.K)
        
    def getAbsoluteScale(self, frame_id):
        """
        specialized for KITTI odometry dataset
        """
        
        gr_pose = self.ground_pose[frame_id-1]
        x_prev = float(gr_pose[0][-1])
        y_prev = float(gr_pose[1][-1])
        z_prev = float(gr_pose[2][-1])
        gr_pose = self.ground_pose[frame_id]
        x = float(gr_pose[0][-1])
        y = float(gr_pose[1][-1])
        z = float(gr_pose[2][-1])
        self.trueX, self.trueY, self.trueZ = x, y, z
        return np.sqrt((x - x_prev)*(x - x_prev) + (y - y_prev)*(y - y_prev) + (z - z_prev)*(z - z_prev))

    def getAbsoluteScaleLoop(self, i, j):
        """
        specialized for KITTI odometry dataset
        """
        
        gr_pose_i = self.ground_pose[i]
        x_prev = float(gr_pose_i[0][-1])
        y_prev = float(gr_pose_i[1][-1])
        z_prev = float(gr_pose_i[2][-1])
        
        gr_pose_j = self.ground_pose[j]
        x = float(gr_pose_j[0][-1])
        y = float(gr_pose_j[1][-1])
        z = float(gr_pose_j[2][-1])
        
        self.trueX, self.trueY, self.trueZ = x, y, z
        return np.sqrt((x - x_prev)*(x - x_prev) + (y - y_prev)*(y - y_prev) + (z - z_prev)*(z - z_prev))
        

    # def run_optimizer(self, local_window=10):

    #     """
    #     Add poses to the optimizer graph
    #     """
    #     if len(self.poses)<local_window+1:
    #         return

    #     # self.pose_graph = PoseGraph(verbose = True)
    #     local_poses = self.poses[1:][-local_window:]

    #     for i in range(1,len(local_poses)):   
    #         self.pose_graph.add_vertex(i, local_poses[i])
    #         self.pose_graph.add_edge((i-1, i), getTransform(local_poses[i], local_poses[i-1]))
    #         self.pose_graph.optimize(self.args.num_iter)
        
    #     self.poses[-local_window+1:] = self.pose_graph.nodes_optimized

    def graph_extend(self, pose, prev_pose, i, j):
        self.pose_graph.add_vertex(i, pose)
        self.pose_graph.add_edge((j, i), getTransform(pose, prev_pose))
        return

    def add_loop_constraint(self, pose, i, j):

        # pose = np.linalg.inv(pose)
        self.pose_graph.add_edge((j, i), pose)
        self.loop_closure_count+=1
        return


    def calculate_errors(self):

        """
        Calculate errors propagated
        """

        error_r , error_t = getError(self.poses[-1], self.poses[-2], self.gt[-1], self.gt[-2])
        self.errors.append((error_r, error_t))

    def model_optimize(self):
        self.pose_graph.optimize(self.args.num_iter)
        self.poses = self.pose_graph.nodes_optimized

    def __call__(self, stage, current_frame):
        
        # wtf stage-1 will read the last pose for the first iteration
        self.gt.append(convert_to_4_by_4(self.ground_pose[stage]))

        self.current_frame = current_frame
        global_flag = False

        if stage == 0:
            """ process first frame """
            self.points_ref = self.detector.detect(current_frame)
            self.points_ref = np.array([x.pt for x in self.points_ref])
            self.prev_frame = current_frame
            self.prev_Rt = convert_to_4_by_4(self.ground_pose[stage])#np.eye(4) 
            self.prev_R = self.prev_Rt[:3,:3]
            self.prev_t = self.prev_Rt[:3, 3].reshape(-1,1)
            # self.poses.append(np.eye(4))
            self.poses.append(self.prev_Rt)
            self.raw.append(self.prev_Rt)

            # wtf check how to set reference pose for the pose graph (set to origin)
            self.pose_graph.add_vertex(stage, self.prev_Rt, True)
            return
    
        elif stage == 1:
            """ process second frame """
            self.points_ref, points_cur = self.feature_tracker(self.prev_frame, current_frame, self.points_ref)
            E, _ = cv2.findEssentialMat(points_cur, self.points_ref, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, __ = cv2.recoverPose(E, points_cur, self.points_ref, self.K)

            self.points_ref = points_cur
            absolute_scale = self.getAbsoluteScale(stage)
            # print(absolute_scale)
            if absolute_scale > 0:
                self.cur_t = self.prev_t + absolute_scale*self.prev_R@t
                self.cur_R = self.prev_R@R

                self.curRaw_R = self.cur_R
                self.curRaw_t = self.cur_t

            self.cur_Rt = convert_to_Rt(self.cur_R, self.cur_t)
            self.poses.append(convert_to_4_by_4(self.cur_Rt))
            self.raw.append(convert_to_4_by_4(self.cur_Rt))
            self.graph_extend(convert_to_4_by_4(self.cur_Rt), self.prev_Rt, stage, stage-1)
        else:
            """ process subsequent frames after first 2 frames """

            self.points_ref, points_cur = self.feature_tracker(self.prev_frame, current_frame, self.points_ref)
            E, mask = cv2.findEssentialMat(points_cur, self.points_ref, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, mask = cv2.recoverPose(E, points_cur, self.points_ref, self.K)
            num_inliers = mask[mask > 0].shape[0]
            # num_outliers = mask.shape[0] - num_inliers
            inlier_ratio = num_inliers/mask.shape[0]
            print("Tracker inlier ratio: ", inlier_ratio)

            if (abs(t[1]) > 0.001):
                t[1] = np.random.uniform(-0.0001, 0.0001, size = 1)
                # print('true')

            absolute_scale = self.getAbsoluteScale(stage)
            # print(absolute_scale)
            if absolute_scale > 0.1:
                self.cur_t = self.prev_t + absolute_scale*self.prev_R@t
                self.cur_R = self.prev_R@R

                self.curRaw_t = self.prevRaw_t + absolute_scale*self.prevRaw_R@t
                self.curRaw_R = self.prevRaw_R@R

            if(self.points_ref.shape[0]<KMIN_NUM_FEATURE):
                points_cur = self.detector.detect(current_frame)
                points_cur = np.array([x.pt for x in points_cur], dtype=np.float32)

            self.points_ref = points_cur
            self.cur_Rt = convert_to_Rt(self.cur_R, self.cur_t)
            self.poses.append(convert_to_4_by_4(self.cur_Rt))
            
            self.curRaw_Rt = convert_to_Rt(self.curRaw_R, self.curRaw_t)
            self.raw.append(convert_to_4_by_4(self.curRaw_Rt))
            
            pose = convert_to_4_by_4(self.cur_Rt)
            prev_pose = convert_to_4_by_4(self.prev_Rt)

            self.graph_extend(pose, prev_pose, stage, stage-1)

            if(stage%5==0):
                found_loop, rel_pose, idx_j = self.loop_closure.check_loop_closure(stage, current_frame)
            else:
                found_loop = False
            if found_loop:
                curr_stage_gt = convert_to_4_by_4(self.ground_pose[stage])
                matched_gt = convert_to_4_by_4(self.ground_pose[idx_j])
                abs_dist = self.getAbsoluteScaleLoop(stage, idx_j)
                if (abs_dist > 0.1 and abs(rel_pose[0,3]*abs_dist) < 1 ):
                    rel_pose[:3,3] = rel_pose[:3,3]*abs_dist
                    self.add_loop_constraint(rel_pose, stage, int(idx_j))
                    print("Found Loop Closure: ", self.loop_closure_count)

                # if(self.loop_closure_count%10 == 0):
                    self.pose_graph.optimize(self.args.num_iter)
                    self.poses = self.pose_graph.nodes_optimized
                    global_flag = True

            # if self.args.optimize:
            #     self.run_optimizer(self.args.local_window)
                #self.calculate_errors()
        #####################################################################################################

        if global_flag:
            last_pose = self.pose_graph.optimizer.vertex(len(self.poses)-1).estimate().matrix()
            self.prev_t = last_pose[:3,3].reshape(-1, 1)
            self.prev_R = last_pose[:3,:3]
        else:
            self.prev_t = self.cur_t 
            self.prev_R = self.cur_R

        self.prevRaw_t = self.curRaw_t
        self.prevRaw_R = self.curRaw_R

        self.prev_Rt = convert_to_Rt(self.prev_R, self.prev_t)        
        if global_flag:
            print("opimized_Rt ", self.prev_Rt)
        self.prev_frame = current_frame