import numpy as np
import open3d as o3d

from pose import Pose
from rigid import get_plane_coordinate_system


class Surface:
    def __init__(self, tag_size):
        self.tag_size = tag_size
        self.tag_poses = []
        self.surface_corners = []
        self.marker_local_coords = None

    def add_tag_pose(self, pose):
        self.tag_poses.append(pose)
        self.surface_corners.append(pose.position)

    def build_surface(self, orient_towards, from_poses=True):
        tag_half = self.tag_size / 2
        tag_plane = np.array(
            [
                [-tag_half, -tag_half, 0],
                [tag_half, -tag_half, 0],
                [tag_half, tag_half, 0],
                [-tag_half, tag_half, 0],
            ]
        )

        points = np.zeros((len(self.tag_poses), 5, 3))
        for i, tag_pose in enumerate(self.tag_poses):
            # Transform tag corners to camera frame using tag_pose
            tag_corners_in_cam = (tag_pose.rotation @ tag_plane.T).T + tag_pose.position

            for c in range(4):
                points[i, c] = tag_corners_in_cam[c]

            points[i, 4] = self.surface_corners[i]

        points.shape = (-1, 3)

        centroid = np.mean(points, axis=0)

        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            plane_model, inliers = pcd.segment_plane(
                distance_threshold=0.01, ransac_n=3, num_iterations=1000
            )
            [a, b, c, d] = plane_model

            inlier_cloud = np.asarray(pcd.select_by_index(inliers).points)

            (self.x_axis, self.y_axis, self.normal) = get_plane_coordinate_system(
                inlier_cloud
            )

            if orient_towards is not None:
                orient_towards = np.asarray(orient_towards, dtype=float)

                ref_vec = orient_towards - centroid.squeeze()

                if np.dot(self.normal, ref_vec) > 0:
                    self.normal = -self.normal

            R = np.zeros((3, 3))
            R[:, 0] = self.x_axis
            R[:, 1] = self.y_axis
            R[:, 2] = self.normal
        except Exception:
            return False

        self.pose_in_neon = Pose(
            position=centroid.flatten(),
            rotation=R,
        )

        return True
