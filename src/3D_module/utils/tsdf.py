import open3d as o3d
from copy import deepcopy

class TSDF:
    def __init__(self, voxel_length: float = 0.001, sdf_trunc: float = 0.1):
        self.tsdf = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_length,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
            volume_unit_resolution=32,
            depth_sampling_stride=8
        )

    def build_3D_map(self, rgbd: o3d.geometry.RGBDImage, intrinsic: o3d.camera.PinholeCameraIntrinsic, extrinsic):
        """
        Reconstruct the 3D model from the pseudo-rgbd using TSDF.
        :param rgbd: pseudo-rgbd
        :param intrinsic: intrinsic parameter of the camera
        :param extrinsic: the global position of the camera
        """
        self.tsdf.integrate(rgbd, intrinsic, extrinsic)

    def build_copy_3D_map(self, rgbd: o3d.geometry.RGBDImage, intrinsic: o3d.camera.PinholeCameraIntrinsic, extrinsic):
        tsdf_copy = deepcopy(self.tsdf)
        tsdf_copy.integrate(rgbd, intrinsic, extrinsic)
        return tsdf_copy

    def save_pcd(self, saving_path: str):
        """
        Save the generated point cloud.
        :param saving_path: path to file
        """
        pcd = self.tsdf.extract_point_cloud()
        o3d.io.write_point_cloud(saving_path, pcd)

    def extract_pcd(self):
        return self.tsdf.extract_point_cloud()

    def extract_mesh(self) -> o3d.geometry.TriangleMesh:
        return self.tsdf.extract_triangle_mesh()

    def save_mesh(self, saving_path: str):
        """
        Save the generated mesh.
        :param saving_path: path to file
        """
        mesh = self.extract_mesh()
        o3d.io.write_triangle_mesh(saving_path, mesh)

