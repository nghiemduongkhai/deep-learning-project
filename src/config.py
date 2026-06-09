import numpy as np
from pathlib import Path


# =========================================================
# Project paths
# =========================================================

# If this file is inside src/config.py, PROJECT_DIR will be the project root.
# If this file is in project_root/config.py, PROJECT_DIR will also be the project root.
_CONFIG_FILE = Path(__file__).resolve()
PROJECT_DIR = _CONFIG_FILE.parent.parent if _CONFIG_FILE.parent.name.lower() == "src" else _CONFIG_FILE.parent

DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "model"
OUTPUT_DIR = PROJECT_DIR / "outputs"
APP_DIR = PROJECT_DIR / "app"

RESNET_PATH = MODEL_DIR / "deeplabv3_resnet50_best.pth"
VIT_PATH = MODEL_DIR / "segformer_b0_best.pth"


# =========================================================
# Dataset
# =========================================================

DATASET = "electraawais/cityscape-dataset"

CITYSCAPES_CLASSES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]

NUM_CLASSES = len(CITYSCAPES_CLASSES)
IGNORE_INDEX = 255


# =========================================================
# Cityscapes labelId -> trainId mapping
# =========================================================

LABELID_TO_TRAINID = {
    0: 255,
    1: 255,
    2: 255,
    3: 255,
    4: 255,
    5: 255,
    6: 255,
    7: 0,     # road
    8: 1,     # sidewalk
    9: 255,
    10: 255,
    11: 2,    # building
    12: 3,    # wall
    13: 4,    # fence
    14: 255,
    15: 255,
    16: 255,
    17: 5,    # pole
    18: 255,
    19: 6,    # traffic light
    20: 7,    # traffic sign
    21: 8,    # vegetation
    22: 9,    # terrain
    23: 10,   # sky
    24: 11,   # person
    25: 12,   # rider
    26: 13,   # car
    27: 14,   # truck
    28: 15,   # bus
    29: 255,
    30: 255,
    31: 16,   # train
    32: 17,   # motorcycle
    33: 18,   # bicycle
    255: 255,
}


# =========================================================
# Visualization colors
# =========================================================

CITYSCAPES_COLORS = np.array([
    [128, 64, 128],    # road
    [244, 35, 232],    # sidewalk
    [70, 70, 70],      # building
    [102, 102, 156],   # wall
    [190, 153, 153],   # fence
    [153, 153, 153],   # pole
    [250, 170, 30],    # traffic light
    [220, 220, 0],     # traffic sign
    [107, 142, 35],    # vegetation
    [152, 251, 152],   # terrain
    [70, 130, 180],    # sky
    [220, 20, 60],     # person
    [255, 0, 0],       # rider
    [0, 0, 142],       # car
    [0, 0, 70],        # truck
    [0, 60, 100],      # bus
    [0, 80, 100],      # train
    [0, 0, 230],       # motorcycle
    [119, 11, 32],     # bicycle
], dtype=np.uint8)


# =========================================================
# Preprocessing
# =========================================================

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# =========================================================
# Training hyperparameters
# =========================================================

BATCH_SIZE = 2
RANDOMSEED = 42
EPOCHS = 10

RESNET_LEARNING_RATE = 1e-4
VIT_LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4


# =========================================================
# Model names
# =========================================================

RESNET_MODEL_NAME = "DeepLabV3-ResNet50"
VIT_MODEL_NAME = "SegFormer-B0"


# =========================================================
# Collision warning / app config
# =========================================================

DANGER_CLASS_NAMES = [
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
]

DANGER_CLASS_IDS = [
    CITYSCAPES_CLASSES.index(name)
    for name in DANGER_CLASS_NAMES
    if name in CITYSCAPES_CLASSES
]

# Risk thresholds based on ratio of danger-class pixels inside ROI.
SAFE_RISK_THRESHOLD = 0.03
WARNING_RISK_THRESHOLD = 0.10

# Trapezoid ROI defaults.
# Larger top_y ratio means the trapezoid starts lower, so ROI is shorter.
ROI_TOP_Y_RATIO = 0.62
ROI_BOTTOM_WIDTH_RATIO = 0.62
ROI_TOP_WIDTH_RATIO = 0.30

# App inference defaults.
APP_DEFAULT_ALPHA = 0.45
APP_INFERENCE_SIZE_OPTIONS = [256, 384, 512]
APP_DEFAULT_INFERENCE_SIZE = 384

APP_MAX_VIDEO_FRAMES = 300
APP_FRAME_STRIDE = 3

APP_VIDEO_MODES = [
    "Both models side-by-side",
    "ResNet only",
    "SegFormer only",
]

# Video encoding settings.
# On Windows, OpenCV may not support H.264/avc1 directly.
# Install imageio-ffmpeg with: pip install imageio-ffmpeg
APP_VIDEO_CODEC = "avc1"
APP_VIDEO_FALLBACK_CODEC = "mp4v"
FFMPEG_PATH = "ffmpeg"
APP_ALLOW_MP4V_FALLBACK = True


# =========================================================
# Utility
# =========================================================

def ensure_project_dirs():
    """Create common project directories if they do not exist."""
    for directory in [DATA_DIR, MODEL_DIR, OUTPUT_DIR, APP_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
