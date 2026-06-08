import numpy as np
from pathlib import Path

DATASET = "electraawais/cityscape-dataset"

CITYSCAPES_CLASSES = ["road", "sidewalk", "building", "wall", "fence",
                      "pole", "traffic light", "traffic sign", "vegetation",
                      "terrain", "sky", "person", "rider", "car",
                      "truck", "bus", "train", "motorcycle", "bicycle"]
NUM_CLASSES = len(CITYSCAPES_CLASSES)

IGNORE_INDEX = 255

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
    255: 255
}

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

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

BATCH_SIZE = 2

RANDOMSEED = 42

EPOCHS = 10

#DeepLabV3-ResNet50
RESNET_LEARNING_RATE = 1e-4
#ViT
VIT_LEARNING_RATE = 5e-5

WEIGHT_DECAY = 1e-4


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "model"

RESNET_PATH = MODEL_DIR / "deeplabv3_resnet50_best.pth"
VIT_PATH = MODEL_DIR / "segformer_b0_best.pth"