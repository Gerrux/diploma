import cv2
import numpy as np

from supervision import BoundingBoxAnnotator, Detections
from supervision.annotators.base import ImageType
from supervision.utils.conversion import convert_for_annotation_method
from supervision.annotators.utils import resolve_color
from ultralytics.engine.results import Keypoints
from ultralytics.utils.plotting import colors


class HumanPoseAnnotator(BoundingBoxAnnotator):
    def __init__(self, *args, keypoint_radius: int = 3, thickness_skeleton: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.limb_color = None
        self.keypoint_radius = keypoint_radius
        self.thickness_skeleton = thickness_skeleton
        self.skeleton = [
            [16, 14],
            [14, 12],
            [17, 15],
            [15, 13],
            [12, 13],
            [6, 12],
            [7, 13],
            [6, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [9, 11],
            [2, 3],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
            [5, 7],
        ]
        self.limb_color = colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
        self.kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]

    def draw_skeleton(self, kpts, shape=(640, 640), radius=5, kpt_line=True, scene=None):
        nkpt, ndim = kpts.shape
        is_pose = nkpt == 17 and ndim in {2, 3}
        kpt_line &= is_pose

        for i, k in enumerate(kpts):
            color_k = [int(x) for x in self.kpt_color[i]] if is_pose else colors(i)
            x_coord, y_coord = k[0], k[1]
            if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                if len(k) == 3:
                    conf = k[2]
                    if conf < 0.5:
                        continue
                cv2.circle(scene, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)

        if kpt_line:
            ndim = kpts.shape[-1]
            for i, sk in enumerate(self.skeleton):
                pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
                pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
                if ndim == 3:
                    conf1 = kpts[(sk[0] - 1), 2]
                    conf2 = kpts[(sk[1] - 1), 2]
                    if conf1 < 0.5 or conf2 < 0.5:
                        continue
                if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                    continue
                if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                    continue
                cv2.line(scene, pos1, pos2, [int(x) for x in self.limb_color[i]], thickness=2, lineType=cv2.LINE_AA)

        return scene

    @convert_for_annotation_method
    def annotate(self, scene: ImageType, detections: Detections, keypoints: Keypoints, **kwargs) -> ImageType:
        scene = super().annotate(scene, detections, **kwargs)

        for detection_idx in range(len(detections)):
            color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=self.color_lookup,
            )

            keypoints_xy = keypoints.xy[detection_idx]
            scene = self.draw_skeleton(keypoints_xy, scene.shape[:2][::-1], kpt_line=True, scene=scene)

        return scene
