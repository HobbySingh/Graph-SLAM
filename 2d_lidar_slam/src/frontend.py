import time

import imageio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import icp
from graph import Graph
from loop_closure import find_loop_closure
from pose_se2 import PoseSE2


def run(clf_file, name, save_gif=True, plot_every=1000):
    if save_gif:
        import atexit

        images = []

        def fn():
            print("Saving GIF")
            imageio.mimsave(f"./../results/slam_{int(time.time())}.gif", images, fps=20)
            print("Done")

        atexit.register(fn)

    with open(clf_file, "r") as f:
        lasers = []
        odoms = []
        for line in f:
            tokens = line.split(" ")
            if tokens[0] == "FLASER":
                num_readings = int(tokens[1])
                scans = np.array(tokens[2 : 2 + num_readings], dtype=np.float)
                scan_time = float(tokens[2 + num_readings + 6])
                index = np.arange(-90, 90 + 180 / num_readings, 180 / num_readings)
                index = np.delete(index, num_readings // 2)
                converted_scans = []
                angles = np.radians(index)
                converted_scans = (
                    np.array([np.cos(angles), np.sin(angles)]).T * scans[:, np.newaxis]
                )
                lasers.append(np.array(converted_scans))
                x = float(tokens[2 + num_readings])
                y = float(tokens[3 + num_readings])
                theta = float(tokens[4 + num_readings])
                odoms.append([x, y, theta])

    odoms = np.array(odoms)
    lasers = np.array(lasers)

    pose_graph = Graph([], [])
    pose = np.eye(3)
    pose_graph.add_vertex(0, PoseSE2.from_rt_matrix(pose))

    init_pose = np.eye(3)
    vertex_idx = 1
    all_lasers = []

    max_x = -float("inf")
    max_y = -float("inf")
    min_x = float("inf")
    min_y = float("inf")

    for od_idx, odom in tqdm(enumerate(odoms[:3700]), total=len(odoms)):
        # Initialize
        if od_idx == 0:
            prev_odom = odom.copy()
            prev_idx = 0
            B = lasers[od_idx]
            all_lasers.append(B)
            continue

        dx = odom - prev_odom

        # IF the robot has moved significantly
        if np.linalg.norm(dx[0:2]) > 0.4 or abs(dx[2]) > 0.2:
            # Scan Matching
            A = lasers[prev_idx]
            B = lasers[od_idx]

            x, y, yaw = dx[0], dx[1], dx[2]
            init_pose = np.array(
                [
                    [np.cos(yaw), -np.sin(yaw), x],
                    [np.sin(yaw), np.cos(yaw), y],
                    [0, 0, 1],
                ]
            )

            with np.errstate(all="raise"):
                try:
                    tran, distances, iter, cov = icp.icp(
                        B, A, init_pose, max_iterations=80, tolerance=0.0001
                    )
                except Exception as e:
                    continue
            init_pose = tran
            pose = pose @ tran
            # breakpoint()
            pose_graph.add_vertex(vertex_idx, PoseSE2.from_rt_matrix(pose))
            information = np.linalg.inv(cov)
            pose_graph.add_edge(
                [vertex_idx - 1, vertex_idx], PoseSE2.from_rt_matrix(tran), information
            )

            prev_odom = odom
            prev_idx = od_idx

            all_lasers.append(B)

            if vertex_idx > 10 and vertex_idx % 10 == 0:
                find_loop_closure(
                    curr_pose=PoseSE2.from_rt_matrix(pose),
                    curr_idx=vertex_idx,
                    laser=all_lasers,
                    g=pose_graph,
                )

                pose_graph.optimize()
                # print(vertex_idx, pose)
                pose = pose_graph.get_rt_matrix(vertex_idx)
                # print(vertex_idx, pose)

            # Draw trajectory and map
            traj = []
            point_cloud = []
            for idx in range(0, vertex_idx):
                # x, y, yaw = pose_graph.get_pose(idx).arr
                # r = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
                # t = np.array([x, y]).T
                mat = pose_graph.get_rt_matrix(idx)
                r = mat[:2, :2]
                t = mat[:2, 2]
                filtered = all_lasers[idx]
                filtered = filtered[np.linalg.norm(filtered, axis=1) < 80]
                point_cloud.append((r @ filtered.T + t[:, np.newaxis]).T)
                traj.append(t[0:2])
            point_cloud = np.vstack(point_cloud)

            xyreso = 0.01  # Map resolution (m)
            point_cloud = (point_cloud / xyreso).astype("int")
            point_cloud = np.unique(point_cloud, axis=0)
            point_cloud = point_cloud * xyreso

            current_max = np.max(point_cloud, axis=0)
            current_min = np.min(point_cloud, axis=0)
            max_x = max(max_x, current_max[0])
            max_y = max(max_y, current_max[1])
            min_x = min(min_x, current_min[0])
            min_y = min(min_y, current_min[1])

            plt.cla()
            plt.axis([min_x, max_x, min_y, max_y])
            # fig = plt.gcf()
            # fig.set_size_inches((9, 9), forward=False)
            traj = np.array(traj)
            plt.plot(traj[:, 0], traj[:, 1], "-g")
            plt.plot(point_cloud[:, 0], point_cloud[:, 1], ".b", markersize=0.1)
            plt.pause(0.0001)

            if save_gif:
                plt.gcf().canvas.draw()
                image = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype="uint8")
                image = image.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
                images.append(image)

            vertex_idx += 1

        # if vertex_idx > 10 and vertex_idx % plot_every == 0:
        #     pose_graph.plot(title=f"{name}_{vertex_idx}")

    return pose_graph
