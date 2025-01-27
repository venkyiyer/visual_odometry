import os
import sys
from utils.slam_utils import *
from utils.posegraph import PoseGraph
from utils.tsdf import TSDF
from utils.visual_odometry import VO

class SLAM:
    def __init__(self, list_of_rgb: list[str], list_of_depth: list[str], path_to_vo_model: str):
        self._initialize_intrinsics()
        self.depth_scale = 1000
        self.perform_loop_closure = False
        self.o3d_device, self.torch_device = device_handler()
        self._initialize_pose_estimation_variables()
        self._initialize_loop_closure_variables()
        self._initialize_main_loop_variables(list_of_rgb, list_of_depth)
        self.map3D = o3d.geometry.PointCloud()
        self.global_posegraph = PoseGraph()
        self.num_posegraph_optim = 500
        self.tsdf = TSDF()
        self.vo = VO(path_to_vo_model, self.o3d_t_intrinsic, self.intrinsics)
        self._initialize_saving_paths()

    def _initialize_intrinsics(self):
        self.o3d_intrinsic, self.o3d_t_intrinsic = get_o3d_intrinsic(
            frame_width=600, frame_height=480, fx=383.1901395, fy=383.1901395, cx=276.4727783203125, cy=124.3335933685303)
        self.intrinsics = np.array([383.1901395, 383.1901395, 276.4727783203125, 124.3335933685303])

    def _initialize_pose_estimation_variables(self):
        self.global_motion = []
        self.inv_global_motion = []
        self.global_extrinsic = []

    def _initialize_loop_closure_variables(self):
        self.num_closure = 10000
        self.global_key_frame_indices = []

    def _initialize_main_loop_variables(self, list_of_rgb, list_of_depth):
        self.list_of_rgb = list_of_rgb
        self.list_of_depth = list_of_depth
        self.n_frames = len(list_of_rgb)

    def _initialize_saving_paths(self):
        self.pcd_save_path = "/home/gvide/Scrivania/slam_test/pcds/pcd_%_.ply"
        self.mesh_save_path = "/home/gvide/Scrivania/slam_test/meshes/mesh_%_.ply"

    def main_loop_no_gui(self):
        for i in range(self.n_frames):
            print(f"[INFO]: Frame {i}/{self.n_frames}")
            if i == 0:
                curr_rgbd, pcd, global_pose = self._first_loop()
                prev_rgbd = None
            else:
                curr_rgbd, prev_rgbd, pcd, global_pose = self._sequential_loop(i)
                print(global_pose)
                if self.perform_loop_closure and i % self.num_closure == 0:
                    self._loop_closure()

    def main_loop_gui(self, i):
        if i == 0:
            return self._first_loop()
        elif i > 0:
            print("hola")
            return self._sequential_loop(i)

    def _first_loop(self):
        initial_motion_matrix = np.identity(4, dtype=np.float64)
        initial_extrinsic_matrix = np.identity(4, dtype=np.float64)
        self._initialize_pose(initial_motion_matrix, initial_extrinsic_matrix)
        self.global_posegraph.add_node(initial_extrinsic_matrix)
        curr_rgbd = self._load_rgbd(0)
        self.tsdf.build_3D_map(curr_rgbd.rgbd_tsdf, self.o3d_intrinsic, initial_extrinsic_matrix)
        pcd = self.tsdf.extract_pcd()
        return curr_rgbd, pcd, initial_extrinsic_matrix

    def _initialize_pose(self, motion_matrix, extrinsic_matrix):
        add_pose_to_list(motion_matrix, self.global_motion)
        add_pose_to_list(motion_matrix, self.inv_global_motion, invert_matrix=True)
        add_pose_to_list(extrinsic_matrix, self.global_extrinsic)

    def _load_rgbd(self, index):
        return RGBD(color_path=self.list_of_rgb[index], depth_path=self.list_of_depth[index], device=self.o3d_device)

    def _sequential_loop(self, i):
        curr_rgbd = self._load_rgbd(i)
        prev_rgbd = self._load_rgbd(i - 1)
        transformation = self.vo.estimate_relative_pose_between(
            prev_frame=self.list_of_rgb[i-1], curr_frame=self.list_of_rgb[i], prev_rgbd=prev_rgbd, curr_rgbd=curr_rgbd, i=i)
        curr_absolute_pose = compute_curr_estimate_global_pose(self.global_extrinsic[-1], transformation)
        self._store_pose(transformation, curr_absolute_pose)
        self.global_posegraph.add_node(curr_absolute_pose)
        self.global_posegraph.add_edge(transformation, i, i - 1, False)
        self._optimize_posegraph_if_needed(i)
        self._integrate_frame(curr_rgbd, curr_absolute_pose, i)
        pcd = self.tsdf.extract_pcd()
        return curr_rgbd, prev_rgbd, pcd, curr_absolute_pose

    def _store_pose(self, transformation, absolute_pose):
        add_pose_to_list(transformation, self.global_motion)
        add_pose_to_list(transformation, self.inv_global_motion, invert_matrix=True)
        add_pose_to_list(absolute_pose, self.global_extrinsic)

    def _optimize_posegraph_if_needed(self, i):
        if i % self.num_posegraph_optim == 0:
            self.global_posegraph.optimize()
            prev_global_extr = self.global_extrinsic
            self.global_extrinsic = update_global_extrinsic(self.global_posegraph.pose_graph)
            if not np.array_equal(prev_global_extr, self.global_extrinsic):
                self.tsdf = update_map_after_pg(self.global_extrinsic, self.list_of_rgb, self.list_of_depth, self.depth_scale, self.o3d_device, self.o3d_intrinsic)

    def _integrate_frame(self, curr_rgbd, curr_absolute_pose, i):
        print("Integrating ...")
        self.tsdf.build_3D_map(curr_rgbd.rgbd_tsdf, self.o3d_intrinsic, curr_absolute_pose)
        if i % 2000 == 0:
            self.tsdf = update_map_after_pg(self.global_extrinsic, self.list_of_rgb, self.list_of_depth, self.depth_scale, self.o3d_device, self.o3d_intrinsic)
        if i == (self.n_frames - 1):
            print("[INFO]: saving mesh and pcd")
            self.tsdf.save_pcd(self.pcd_save_path.replace("%", str(i)))
            self.tsdf.save_mesh(self.mesh_save_path.replace("%", str(i)))
            sys.exit()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SLAM system")
    parser.add_argument("--depth_map_path", type=str, required=True, help="Path to the depth map directory")
    parser.add_argument("--rgb_path", type=str, required=True, help="Path to the RGB images directory")
    parser.add_argument("--path_to_model", type=str, required=True, help="Path to the visual odometry model")
    args = parser.parse_args()

    rgb_list = sorted([os.path.join(args.rgb_path, f) for f in os.listdir(args.rgb_path)])
    depth_list = sorted([os.path.join(args.depth_map_path, f) for f in os.listdir(args.depth_map_path)])
    slam = SLAM(rgb_list, depth_list, args.path_to_model)
    slam.main_loop_no_gui()