# 2D Lidar SLAM

In this we implement custom frontend and backend modules. Backend consists of custom 2D Pose Graph construction and optimization implementation. Frontend uses odometry, lidar scan and scan matching to compute graph poses. Loop closure constraints are added using ICP. We first implemented (main_g2o.py) the backend graph optimization using the processed g2o data from [Luca Carlone datasets](https://lucacarlone.mit.edu/datasets/). Once this was achieved, we moved on to using the raw data provided by [Intel Research Lab](http://ais.informatik.uni-freiburg.de/slamevaluation/datasets.php) and building our own pose graph from the laser and odometry information.

## Setup and Usage

```
conda create -n graph_slam python=3.6
conda activate graph_slam
pip install -r requirements.txt

python main_clf.py ../data/intel.clf
```

Pose Graph before optimization |  Pose Graph after optimization
:-------------------------:|:-------------------------:
![](results/graph_optimization/Before_INTEL.png)  |  ![](results/graph_optimization/After_INTEL.png)

Real-Time Pose Graph Generation and Optimization after Loop Closures
:-------------------------:|
![](results/slam_intel_3700.gif)
