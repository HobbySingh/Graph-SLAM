import numpy as np 
import argparse
import sys
sys.path.append("./thirdparty/g2opy/lib/")
sys.path.append("./thirdparty/pangolin/")
    
from core.model import VisualSLAM
from core.dataset import KittiDataset
from core.utils import draw_trajectory
from core.display2D import Displayer
from core.display3D import Viewer3D

def parse_argument():
    
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset', default='kitti')
    parser.add_argument('--path', required=True)
    parser.add_argument('--gt_loops', required=True)
    parser.add_argument('--optimize', action='store_true', help='enable pose graph optimization')
    parser.add_argument('--local_window', default=10, type=int, help='number of frames to run the optimization')
    parser.add_argument('--num_iter', default=300, type=int, help='number of max iterations to run the optimization')
    
    return parser.parse_args()

def main():
    
    args = parse_argument()
    
    # Get data params using the dataloader 
    dataset = KittiDataset(args.path)
    camera_matrix = dataset.intrinsic
    ground_truth_poses = dataset.ground_truth
    num_frames = len(dataset)

    # Initialise the mono-VO model
    model = VisualSLAM(camera_matrix, ground_truth_poses, dataset, args)
    
    # Initialie the viewer object
    viewer = Viewer3D()
    
    # Iterate over the frames and update the rotation and translation vectors
    for index in range(0, num_frames):

        print("Number of frames remaining: ", num_frames-index)

        frame, _ , _ = dataset[index]
        model(index, frame)
        
        if(index == int(num_frames)-10):
            model.model_optimize()

        if index>2:
            viewer.update(model)


    viewer.update(model)
    viewer.stop()

    optimized_poses = [model.pose_graph.optimizer.vertex(i).estimate().matrix() for i in range(len(model.poses))]
    np.save('./data/'+args.path[-3:-1]+'Poses', optimized_poses)
    np.save('./data/'+args.path[-3:-1]+'Gt', np.array(model.gt))
    np.save('./data/'+args.path[-3:-1]+'Raw', np.array(model.raw))


if __name__ == "__main__":
    main()