import cv2
import config
import threading
from threading import Lock
from src.action_recognition.push_up_recognizer import PushUpRecognizer
from src.counter.keypoint_based_counter import KeypointBasedCounter
from src.keypoint_detection.human_pose_detector import DaisykitHumanPoseDetector
from src.keypoint_detection.tracker import KeypointTracker
from src.utils.ui_drawer import UIDrawer
from src.utils.video_grabber import VideoGrabber
from src.utils.visualizer import visualize_keypoints

video_grabber = VideoGrabber(config.DEFAULT_VIDEO_PATH, max_width=512).start()
pushup_counter = KeypointBasedCounter()
ui_drawer = UIDrawer(pushup_counter)
keypoint_detector = DaisykitHumanPoseDetector(config.DAISYKIT_HUMAN_POSE_FLOW_CONFIG)
pushup_recognizer = PushUpRecognizer(config.ACTION_RECOGNITION_MODEL_PATH)
tracker = KeypointTracker()

global_points = []
keypoint_lock = Lock()
is_pushup_score = 1.0

APP_NAME = "PushUp App V2"
cv2.namedWindow(APP_NAME)

# Keypoint detection thread
def keypoint_thread(video_grabber, keypoint_detector):
    global global_points
    while True:
        if video_grabber is not None:
            frame = video_grabber.get_frame()
            points = keypoint_detector.detect(frame, threshold=0.3)
            keypoint_lock.acquire()
            global_points = points
            keypoint_lock.release()
keypoint_t = threading.Thread(target=keypoint_thread, args=(video_grabber, keypoint_detector))
keypoint_t.daemon = True
keypoint_t.start()


# Action recognition thread
# def action_recognition_thread(video_grabber):
#     global is_pushup_score
#     while True:
#         if video_grabber is not None:
#             frame = video_grabber.get_frame()
#             is_pushup_score, _  = pushup_recognizer.update_frame(frame, return_raw_score=True)
# action_recognition_t = threading.Thread(target=action_recognition_thread, args=(video_grabber,))
# action_recognition_t.daemon = True
# action_recognition_t.start()


# Main loop
while True:
    video_frame = video_grabber.get_frame()
    keypoint_lock.acquire()
    points = global_points
    keypoint_lock.release()

    # Update counter
    pushup_counter.update_points(points)
    is_pushup = is_pushup_score > config.PUSHUP_THRESH
    pushup_counter.set_counting(is_pushup)

    # Draw frame
    video_frame = visualize_keypoints(video_frame, points,
        point_color=(0,0,255), text_color=(0,255,0))
    text = "Pushing {}".format(is_pushup_score)
    color = (0, 255, 0)
    if not is_pushup:
        text = "Not Pushing {}".format(is_pushup_score)
        color = (0, 0, 255)
    cv2.putText(video_frame, text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX,  
                0.5, color, 1, cv2.LINE_AA) 
    ui_drawer.set_frame(video_frame)
    draw = ui_drawer.render()

    cv2.imshow(APP_NAME, draw)
    k = cv2.waitKey(10)
    k = k & 0xFF
    if k == ord("o"):
        video_grabber.choose_new_file()
        keypoint_lock.acquire()
        global_points = []
        keypoint_lock.release()
        pushup_counter.reset()
    elif k == ord("c"):
        video_grabber.open_camera()
        keypoint_lock.acquire()
        global_points = []
        keypoint_lock.release()
        pushup_counter.reset()
    elif k == ord("q"):
        exit(0)
