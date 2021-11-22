import cv2
import json
from daisykit.utils import get_asset_file, to_py_type
from daisykit import HumanPoseMoveNetFlow


class DaisykitHumanPoseDetector():
    """Daisykit Human Pose Detector
    Detect multiple human poses from image 
    """

    def __init__(self, config):
        self.human_pose_flow = HumanPoseMoveNetFlow(json.dumps(config))

    def detect(self, img, threshold=0.3, debug=False):
        """Detect human poses
        """

        # Convert image to RGB color space before pushing into Daisykit
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect pose
        poses = self.human_pose_flow.Process(img)

        if debug:
            draw = img.copy()
            self.human_pose_flow.DrawResult(draw, poses)
            draw = cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)
            cv2.imshow("Pose Result", draw)
            cv2.waitKey(1)

        # Convert poses to Python list of dict
        poses = to_py_type(poses)

        # Currently we only support single human pose for pushup counter
        # => Filter the largest object
        keypoints = []
        max_area = -1
        largest_pose = None
        for pose in poses:
            area = pose["w"] * pose["h"]
            if area > max_area:
                max_area = area
                largest_pose = pose
        if largest_pose is not None:
            keypoints = [[p["x"], p["y"], 1 if p["confidence"] >= threshold else 0] for p in pose["keypoints"]]

        return keypoints