import numpy as np
import open3d as o3d

from surface import Surface
from apriltags import AprilTags
from pose import Pose
from rigid import get_plane_coordinate_system


class MocapIRMarker:
    """
    Holds the timeseries of positions of an IR Marker
    """

    def __init__(self, Xs, Ys, Zs, id):
        self.Xs = Xs
        self.Ys = Ys
        self.Zs = Zs

        self.position = np.array([Xs, Ys, Zs])

        self.id = id


class MocapHead:
    def __init__(self):
        self.markers = []

    def add_marker(self, marker):
        self.markers.append(marker)


class MocapAprilTag:
    def __init__(self, tag_id):
        self.markers = []
        self.center = np.array([0, 0, 0])
        self.tag_id = tag_id

    def add_marker(self, marker):
        self.markers.append(marker)

    def estimate_tag_center(self):
        pos = np.array(
            [
                [marker.Xs for marker in self.markers],
                [marker.Ys for marker in self.markers],
                [marker.Zs for marker in self.markers],
            ]
        )
        self.center = np.mean(pos, axis=1)

    def estimate_size_mm(self):
        """
        Estimate tag size [m] from mocap data.
        """

        tag_marker1_pos = np.array(
            [
                self.markers[0].Xs,
                self.markers[0].Ys,
                self.markers[0].Zs,
            ]
        )
        tag_marker2_pos = np.array(
            [
                self.markers[1].Xs,
                self.markers[1].Ys,
                self.markers[1].Zs,
            ]
        )

        self.tag_size = np.sqrt(
            np.sum(
                np.power(
                    tag_marker1_pos - tag_marker2_pos,
                    2,
                )
            )
        )


class MocapSurface:
    def __init__(self):
        self.apriltags = []

    def add_apriltag(self, apriltag):
        self.apriltags.append(apriltag)

    def construct_pose(self, orient_towards=None):
        """
        Construct the estimated pose of the surface in mocap system.
        """

        apriltag = []
        xs, ys, zs = [], [], []
        for apriltag in self.apriltags:
            for marker in apriltag.markers:
                xs.append(marker.Xs)
                ys.append(marker.Ys)
                zs.append(marker.Zs)

        poses = np.vstack([xs, ys, zs]).T
        # print(poses.shape)
        # centroid, rotation = fit_plane(poses)
        # poses.shape = (-1, 3)

        centroid = np.mean(poses, axis=0)
        # print(centroid)

        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(poses)

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

        self.pose = Pose(
            position=centroid.flatten(),
            rotation=R,
        )

        apriltag.estimate_size_mm()
        self.tag_size = apriltag.tag_size


def unify_calib_data(neon, mocap_surface, img, R_apriltag_to_mocap):
    # detect apriltags in neon image
    neon_apriltags = AprilTags(neon, mocap_surface.tag_size, img)

    # find neon's pose in each apriltag coordinate system
    for pose in neon_apriltags.tag_poses:
        neon.add_pose_in_tag(pose.inverse())

    # take detected tag poses and combine them into a surface
    neon_surface = Surface(mocap_surface.tag_size)
    for pose in neon_apriltags.tag_poses:
        neon_surface.add_tag_pose(pose)

    # build the surface from the tags
    ok = neon_surface.build_surface(
        orient_towards=neon.pose_in_tags[0].position, from_poses=True
    )

    if not ok:
        return None, None

    # find neon's pose in local surface coordinate system
    # NOTE: the local surface coordinate system follows the conventions of
    # our SVD method in `fit_plane`.
    # It does not follow the MoCap coordinate system conventions, but that is okay,
    # as the end result is in the MoCap coordinate system.
    neon.set_pose_in_surface(neon_surface.pose_in_neon.inverse())

    # apply surface pose in mocap sys to neon pose in surface coordinates
    # to get neon camera pose in mocap coordinates
    neon.calculate_pose_in_mocap(mocap_surface.pose, R_apriltag_to_mocap)

    return neon_surface, neon_apriltags
