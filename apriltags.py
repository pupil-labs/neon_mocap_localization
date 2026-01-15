import cv2
import numpy as np

from pose import Pose


class AprilTags:
    def __init__(self, detector, neon, tag_size, img):
        self.detector = detector
        self.K = neon.camera_matrix
        self.D = neon.dist_coeffs
        self.tag_size = tag_size

        self.good_detection = True
        self.reprojection_errors = []

        self.undist_frame = None
        self.tag_poses = []
        self.tag_corners = []
        self.tag_ids = []

        self.detect_tags(img)
        if self.good_detection:
            self.extract_tag_poses()

    def detect_tags(self, image):
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        self.new_K, _ = cv2.getOptimalNewCameraMatrix(
            self.K, self.D, img_gray.shape[::-1], 1, img_gray.shape[::-1]
        )
        self.undist_frame = cv2.undistort(
            img_gray, self.K, self.D, None, newCameraMatrix=self.new_K
        )

        fx = self.new_K[0, 0]
        fy = self.new_K[1, 1]
        cx = self.new_K[0, 2]
        cy = self.new_K[1, 2]

        try:
            self.at_detection = self.detector.detect(
                self.undist_frame,
                estimate_tag_pose=True,
                tag_size=self.tag_size,
                camera_params=[fx, fy, cx, cy],
            )
        except Exception:
            self.good_detection = False
            return

        if len(self.at_detection) < 4:
            self.good_detection = False
            return

        tag_points_3d = np.array(
            [
                [self.tag_size / 2, -self.tag_size / 2, 0],  # TR
                [-self.tag_size / 2, -self.tag_size / 2, 0],  # TL
                [-self.tag_size / 2, self.tag_size / 2, 0],  # BL
                [self.tag_size / 2, self.tag_size / 2, 0],  # BR
            ]
        )

        zeroed_D = np.zeros((5, 1), dtype=np.float32)

        for detection in self.at_detection:
            # SOLVEPNP_IPPE_SQUARE returns 2 solutions for rotation/position/error.
            # First one always has smallest error
            ok, (tag_rotation, _), (tag_position, _), (error, _) = cv2.solvePnPGeneric(
                objectPoints=tag_points_3d,
                imagePoints=detection.corners,
                cameraMatrix=self.new_K,
                distCoeffs=zeroed_D,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )

            if not ok:
                self.good_detection = False
                return

            self.reprojection_errors.append(error)

            # 1 is BL
            # 2 is BR
            # 3 is TR
            # 4 is TL
            self.tag_corners.append(detection.corners)
            self.tag_ids.append(detection.tag_id)

    def extract_tag_poses(self):
        for detection in self.at_detection:
            tag_pose = Pose(
                position=detection.pose_t.flatten(),
                rotation=detection.pose_R,
            )

            self.tag_poses.append(tag_pose)
