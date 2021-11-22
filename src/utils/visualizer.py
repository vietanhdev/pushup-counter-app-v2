import cv2
import numpy as np

def visualize_keypoints(image, keypoints, point_color=(0, 255, 0), text_color=(0, 0, 0)):
    """Visualize keypoints

    Args:
        image: Input image
        keypoints: Keypoints in format [[x1, y1], [x2, y2], ...]
        visibility [list]: List of visibilities of keypoints. 0: occluded, 1: visible

    Returns:
        Visualized image
    """

    draw = image.copy()
    for i, p in enumerate(keypoints):
        x, y, visibility = p
        if visibility:
            tmp_point_color = point_color
            draw = cv2.circle(draw, center=(int(x), int(y)),
                            color=tmp_point_color, radius=5, thickness=-1)
            draw = cv2.putText(draw, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, text_color, 1, cv2.LINE_AA)
    return draw