from daisykit.utils import get_asset_file

DEFAULT_VIDEO_PATH = 0 # 0 for webcam, or path to video file

DAISYKIT_HUMAN_POSE_FLOW_CONFIG = config = {
    "person_detection_model": {
        "model": get_asset_file("models/human_detection/ssd_mobilenetv2.param"),
        "weights": get_asset_file("models/human_detection/ssd_mobilenetv2.bin"),
        "input_width": 320,
        "input_height": 320,
        "use_gpu": False
    },
    "human_pose_model": {
        "model": get_asset_file("models/human_pose_detection/movenet/lightning.param"),
        "weights": get_asset_file("models/human_pose_detection/movenet/lightning.bin"),
        "input_width": 192,
        "input_height": 192,
        "use_gpu": False
    }
}

ACTION_RECOGNITION_MODEL_PATH = "trained_models/action-recognition-mobilenetv2-2020-12-14.h5"

PUSHUP_THRESH = 0.9
