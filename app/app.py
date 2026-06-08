import os
import re
import tempfile
import importlib.util
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50
from transformers import SegformerForSemanticSegmentation


# =========================================================
# Load config.py directly by file path
# =========================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent if APP_DIR.name.lower() == "app" else APP_DIR

CONFIG_CANDIDATES = [
    PROJECT_DIR / "src" / "config.py",
    PROJECT_DIR / "config.py",
    APP_DIR / "src" / "config.py",
]

CONFIG_PATH = None
for candidate in CONFIG_CANDIDATES:
    if candidate.exists():
        CONFIG_PATH = candidate
        break

if CONFIG_PATH is None:
    raise FileNotFoundError(
        "Cannot find config.py. Put it in project_root/src/config.py or project_root/config.py."
    )

spec = importlib.util.spec_from_file_location("dl_project_config", CONFIG_PATH)
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)


# =========================================================
# Config values
# =========================================================

CITYSCAPES_CLASSES = cfg.CITYSCAPES_CLASSES
NUM_CLASSES = cfg.NUM_CLASSES
CITYSCAPES_COLORS = cfg.CITYSCAPES_COLORS
IGNORE_INDEX = getattr(cfg, "IGNORE_INDEX", 255)

IMAGENET_MEAN = np.array(
    getattr(cfg, "IMAGENET_MEAN", [0.485, 0.456, 0.406]),
    dtype=np.float32,
)
IMAGENET_STD = np.array(
    getattr(cfg, "IMAGENET_STD", [0.229, 0.224, 0.225]),
    dtype=np.float32,
)

RESNET_MODEL_NAME = getattr(cfg, "RESNET_MODEL_NAME", "DeepLabV3-ResNet50")
VIT_MODEL_NAME = getattr(cfg, "VIT_MODEL_NAME", "SegFormer-B0")

DANGER_CLASS_NAMES = getattr(
    cfg,
    "DANGER_CLASS_NAMES",
    ["person", "rider", "car", "truck", "bus", "motorcycle", "bicycle"],
)

DANGER_CLASS_IDS = getattr(
    cfg,
    "DANGER_CLASS_IDS",
    [
        CITYSCAPES_CLASSES.index(name)
        for name in DANGER_CLASS_NAMES
        if name in CITYSCAPES_CLASSES
    ],
)

SAFE_RISK_THRESHOLD = float(getattr(cfg, "SAFE_RISK_THRESHOLD", 0.03))
WARNING_RISK_THRESHOLD = float(getattr(cfg, "WARNING_RISK_THRESHOLD", 0.10))

ROI_TOP_Y_RATIO = float(getattr(cfg, "ROI_TOP_Y_RATIO", 0.62))
ROI_BOTTOM_WIDTH_RATIO = float(getattr(cfg, "ROI_BOTTOM_WIDTH_RATIO", 0.62))
ROI_TOP_WIDTH_RATIO = float(getattr(cfg, "ROI_TOP_WIDTH_RATIO", 0.30))

APP_DEFAULT_ALPHA = float(getattr(cfg, "APP_DEFAULT_ALPHA", 0.45))
APP_INFERENCE_SIZE_OPTIONS = list(getattr(cfg, "APP_INFERENCE_SIZE_OPTIONS", [256, 384, 512]))
APP_DEFAULT_INFERENCE_SIZE = int(getattr(cfg, "APP_DEFAULT_INFERENCE_SIZE", 384))
APP_MAX_VIDEO_FRAMES = int(getattr(cfg, "APP_MAX_VIDEO_FRAMES", 300))
APP_FRAME_STRIDE = int(getattr(cfg, "APP_FRAME_STRIDE", 3))
APP_VIDEO_MODES = list(
    getattr(
        cfg,
        "APP_VIDEO_MODES",
        ["Both models side-by-side", "ResNet only", "SegFormer only"],
    )
)


def resolve_checkpoint_path(config_attr: str, fallback_filename: str) -> Path:
    raw_path = getattr(cfg, config_attr, None)

    if raw_path is not None:
        path = Path(raw_path)
        if path.exists():
            return path

    candidates = [
        PROJECT_DIR / "model" / fallback_filename,
        PROJECT_DIR / "Model" / fallback_filename,
        APP_DIR / "model" / fallback_filename,
        APP_DIR / "Model" / fallback_filename,
    ]

    for path in candidates:
        if path.exists():
            return path

    return Path(raw_path) if raw_path is not None else candidates[0]


DEFAULT_RESNET_PATH = resolve_checkpoint_path(
    "RESNET_PATH",
    "deeplabv3_resnet50_best.pth",
)
DEFAULT_VIT_PATH = resolve_checkpoint_path(
    "VIT_PATH",
    "segformer_b0_best.pth",
)


# =========================================================
# Utility functions
# =========================================================

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def decode_segmentation_mask(mask: np.ndarray) -> np.ndarray:
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for class_id in range(NUM_CLASSES):
        color_mask[mask == class_id] = CITYSCAPES_COLORS[class_id]

    return color_mask


def preprocess_image(image_rgb: np.ndarray, infer_height: int, infer_width: int):
    original_h, original_w = image_rgb.shape[:2]

    resized = cv2.resize(
        image_rgb,
        (infer_width, infer_height),
        interpolation=cv2.INTER_LINEAR,
    )

    image_float = resized.astype(np.float32) / 255.0
    image_float = (image_float - IMAGENET_MEAN) / IMAGENET_STD

    tensor = torch.from_numpy(image_float).permute(2, 0, 1).float().unsqueeze(0)

    return tensor, (original_h, original_w)


def create_danger_roi(
    height: int,
    width: int,
    top_y_ratio: float = ROI_TOP_Y_RATIO,
    bottom_width_ratio: float = ROI_BOTTOM_WIDTH_RATIO,
    top_width_ratio: float = ROI_TOP_WIDTH_RATIO,
) -> np.ndarray:
    roi = np.zeros((height, width), dtype=np.uint8)

    center_x = 0.50
    bottom_left_x = int((center_x - bottom_width_ratio / 2) * width)
    bottom_right_x = int((center_x + bottom_width_ratio / 2) * width)
    top_left_x = int((center_x - top_width_ratio / 2) * width)
    top_right_x = int((center_x + top_width_ratio / 2) * width)
    top_y = int(top_y_ratio * height)

    polygon = np.array([[
        (bottom_left_x, height - 1),
        (bottom_right_x, height - 1),
        (top_right_x, top_y),
        (top_left_x, top_y),
    ]], dtype=np.int32)

    cv2.fillPoly(roi, polygon, 1)
    return roi


def compute_collision_risk(
    pred_mask: np.ndarray,
    top_y_ratio: float = ROI_TOP_Y_RATIO,
    bottom_width_ratio: float = ROI_BOTTOM_WIDTH_RATIO,
    top_width_ratio: float = ROI_TOP_WIDTH_RATIO,
):
    h, w = pred_mask.shape

    roi = create_danger_roi(
        height=h,
        width=w,
        top_y_ratio=top_y_ratio,
        bottom_width_ratio=bottom_width_ratio,
        top_width_ratio=top_width_ratio,
    )

    danger_mask = np.isin(pred_mask, DANGER_CLASS_IDS).astype(np.uint8)
    overlap = danger_mask * roi

    roi_area = roi.sum() + 1e-6
    overlap_ratio = float(overlap.sum() / roi_area)

    if overlap_ratio < SAFE_RISK_THRESHOLD:
        status = "SAFE"
    elif overlap_ratio < WARNING_RISK_THRESHOLD:
        status = "WARNING"
    else:
        status = "DANGER"

    return status, overlap_ratio, roi


def draw_roi_on_image(image_rgb: np.ndarray, roi: np.ndarray) -> np.ndarray:
    output_bgr = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    contours, _ = cv2.findContours(
        roi.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(output_bgr, contours, -1, (0, 255, 255), 3)
    return cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)


def overlay_mask(image_rgb: np.ndarray, pred_mask: np.ndarray, alpha: float = APP_DEFAULT_ALPHA) -> np.ndarray:
    color_mask = decode_segmentation_mask(pred_mask)

    if color_mask.shape[:2] != image_rgb.shape[:2]:
        color_mask = cv2.resize(
            color_mask,
            (image_rgb.shape[1], image_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    return cv2.addWeighted(image_rgb, 1 - alpha, color_mask, alpha, 0)


def add_status_text(image_rgb: np.ndarray, title: str, status: str, risk_score: float) -> np.ndarray:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if status == "SAFE":
        color = (0, 200, 0)
    elif status == "WARNING":
        color = (0, 200, 255)
    else:
        color = (0, 0, 255)

    text = f"{title}: {status} | risk={risk_score:.3f}"

    cv2.rectangle(image_bgr, (15, 15), (620, 70), (0, 0, 0), -1)
    cv2.putText(
        image_bgr,
        text,
        (25, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        color,
        2,
        cv2.LINE_AA,
    )

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def add_panel_title(image_rgb: np.ndarray, title: str) -> np.ndarray:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(image_bgr, (15, 15), (460, 60), (0, 0, 0), -1)
    cv2.putText(
        image_bgr,
        title,
        (25, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


# =========================================================
# Model builders
# =========================================================

def build_deeplabv3_resnet50(num_classes: int):
    model = deeplabv3_resnet50(weights=None, aux_loss=True)

    model.classifier[-1] = nn.Conv2d(
        in_channels=256,
        out_channels=num_classes,
        kernel_size=1,
    )

    if model.aux_classifier is not None:
        model.aux_classifier[-1] = nn.Conv2d(
            in_channels=256,
            out_channels=num_classes,
            kernel_size=1,
        )

    return model


def build_segformer_b0(num_classes: int):
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    model.config.num_labels = num_classes
    return model


# =========================================================
# SegFormer state_dict compatibility
# =========================================================

def convert_old_segformer_key(key: str) -> str:
    """Convert older HF SegFormer key names to newer HF key names."""

    match = re.match(r"^segformer\.stages\.(\d+)\.patch_embeddings\.(.+)$", key)
    if match:
        stage, rest = match.groups()
        return f"segformer.encoder.patch_embeddings.{stage}.{rest}"

    match = re.match(r"^segformer\.stages\.(\d+)\.layer_norm\.(.+)$", key)
    if match:
        stage, rest = match.groups()
        return f"segformer.encoder.layer_norm.{stage}.{rest}"

    match = re.match(r"^segformer\.stages\.(\d+)\.blocks\.(\d+)\.(.+)$", key)
    if match:
        stage, block, rest = match.groups()

        replacements = [
            ("layernorm_before", "layer_norm_1"),
            ("layernorm_after", "layer_norm_2"),
            ("attention.q_proj", "attention.self.query"),
            ("attention.k_proj", "attention.self.key"),
            ("attention.v_proj", "attention.self.value"),
            ("attention.o_proj", "attention.output.dense"),
            ("attention.sequence_reduction.sequence_reduction", "attention.self.sr"),
            ("attention.sequence_reduction.layer_norm", "attention.self.layer_norm"),
            ("mlp.fc1", "mlp.dense1"),
            ("mlp.fc2", "mlp.dense2"),
        ]

        for old, new in replacements:
            if rest.startswith(old):
                rest = rest.replace(old, new, 1)
                break

        return f"segformer.encoder.block.{stage}.{block}.{rest}"

    match = re.match(r"^decode_head\.linear_projections\.(\d+)\.proj\.(.+)$", key)
    if match:
        index, rest = match.groups()
        return f"decode_head.linear_c.{index}.proj.{rest}"

    return key


def convert_segformer_state_dict_if_needed(state_dict):
    converted = OrderedDict()
    changed = False

    for key, value in state_dict.items():
        new_key = convert_old_segformer_key(key)
        if new_key != key:
            changed = True
        converted[new_key] = value

    return converted, changed


def load_state_dict_with_segformer_compat(model, state_dict, model_name: str):
    if model_name != VIT_MODEL_NAME:
        model.load_state_dict(state_dict, strict=True)
        return

    try:
        model.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError as original_error:
        converted_state_dict, changed = convert_segformer_state_dict_if_needed(state_dict)

        if not changed:
            raise original_error

        try:
            model.load_state_dict(converted_state_dict, strict=True)

        except RuntimeError:
            raise original_error


@st.cache_resource
def load_model(model_name: str, checkpoint_path: str, device_name: str):
    device = torch.device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    num_classes = checkpoint.get("num_classes", NUM_CLASSES)
    checkpoint_model_name = checkpoint.get("model_name", "UNKNOWN")

    if model_name == RESNET_MODEL_NAME:
        model = build_deeplabv3_resnet50(num_classes)
    elif model_name == VIT_MODEL_NAME:
        model = build_segformer_b0(num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    try:
        load_state_dict_with_segformer_compat(
            model=model,
            state_dict=checkpoint["model"],
            model_name=model_name,
        )
    except RuntimeError as error:
        st.error(f"Cannot load checkpoint for {model_name}.")
        st.write("Checkpoint path:", checkpoint_path)
        st.write("Checkpoint model_name:", checkpoint_model_name)
        st.write("Expected model:", model_name)
        st.text(str(error))
        raise error

    model = model.to(device)
    model.eval()

    return model


# =========================================================
# Prediction
# =========================================================

def predict_mask(
    model,
    model_name: str,
    image_rgb: np.ndarray,
    device,
    infer_height: int,
    infer_width: int,
):
    input_tensor, original_size = preprocess_image(
        image_rgb=image_rgb,
        infer_height=infer_height,
        infer_width=infer_width,
    )
    input_tensor = input_tensor.to(device)

    with torch.inference_mode():
        if model_name == RESNET_MODEL_NAME:
            outputs = model(input_tensor)
            logits = outputs["out"]
        elif model_name == VIT_MODEL_NAME:
            outputs = model(pixel_values=input_tensor)
            logits = outputs.logits
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        logits = F.interpolate(
            logits,
            size=(infer_height, infer_width),
            mode="bilinear",
            align_corners=False,
        )

        pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    original_h, original_w = original_size

    pred_mask = cv2.resize(
        pred_mask,
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST,
    )

    return pred_mask


def process_image(
    model,
    model_name: str,
    display_name: str,
    image_rgb: np.ndarray,
    device,
    alpha: float,
    infer_height: int,
    infer_width: int,
    roi_top_y: float,
    roi_bottom_width: float,
    roi_top_width: float,
):
    pred_mask = predict_mask(
        model=model,
        model_name=model_name,
        image_rgb=image_rgb,
        device=device,
        infer_height=infer_height,
        infer_width=infer_width,
    )

    status, risk_score, roi = compute_collision_risk(
        pred_mask=pred_mask,
        top_y_ratio=roi_top_y,
        bottom_width_ratio=roi_bottom_width,
        top_width_ratio=roi_top_width,
    )

    mask_color = decode_segmentation_mask(pred_mask)
    overlay = overlay_mask(image_rgb, pred_mask, alpha=alpha)
    overlay_roi = draw_roi_on_image(overlay, roi)
    final_image = add_status_text(overlay_roi, display_name, status, risk_score)

    return {
        "pred_mask": pred_mask,
        "mask_color": mask_color,
        "overlay": overlay,
        "final_image": final_image,
        "status": status,
        "risk_score": risk_score,
        "roi": roi,
    }


def hstack_same_height(left_rgb: np.ndarray, right_rgb: np.ndarray) -> np.ndarray:
    h = min(left_rgb.shape[0], right_rgb.shape[0])
    left_w = int(left_rgb.shape[1] * h / left_rgb.shape[0])
    right_w = int(right_rgb.shape[1] * h / right_rgb.shape[0])

    left = cv2.resize(left_rgb, (left_w, h), interpolation=cv2.INTER_AREA)
    right = cv2.resize(right_rgb, (right_w, h), interpolation=cv2.INTER_AREA)

    return np.concatenate([left, right], axis=1)


def process_two_models_image(
    resnet_model,
    segformer_model,
    image_rgb: np.ndarray,
    device,
    alpha: float,
    infer_height: int,
    infer_width: int,
    roi_top_y: float,
    roi_bottom_width: float,
    roi_top_width: float,
):
    resnet_result = process_image(
        model=resnet_model,
        model_name=RESNET_MODEL_NAME,
        display_name="ResNet",
        image_rgb=image_rgb,
        device=device,
        alpha=alpha,
        infer_height=infer_height,
        infer_width=infer_width,
        roi_top_y=roi_top_y,
        roi_bottom_width=roi_bottom_width,
        roi_top_width=roi_top_width,
    )

    segformer_result = process_image(
        model=segformer_model,
        model_name=VIT_MODEL_NAME,
        display_name="SegFormer",
        image_rgb=image_rgb,
        device=device,
        alpha=alpha,
        infer_height=infer_height,
        infer_width=infer_width,
        roi_top_y=roi_top_y,
        roi_bottom_width=roi_bottom_width,
        roi_top_width=roi_top_width,
    )

    comparison = hstack_same_height(
        add_panel_title(resnet_result["final_image"], RESNET_MODEL_NAME),
        add_panel_title(segformer_result["final_image"], VIT_MODEL_NAME),
    )

    return resnet_result, segformer_result, comparison


# =========================================================
# Video
# =========================================================

def process_video_compare(
    resnet_model,
    segformer_model,
    input_video_path: str,
    output_video_path: str,
    device,
    alpha: float,
    max_input_frames: int,
    frame_stride: int,
    infer_height: int,
    infer_width: int,
    roi_top_y: float,
    roi_bottom_width: float,
    roi_top_width: float,
    video_mode: str,
):
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        raise RuntimeError("Cannot open uploaded video.")

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if input_fps <= 0 or np.isnan(input_fps):
        input_fps = 20

    output_fps = max(input_fps / max(frame_stride, 1), 1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_width = width * 2 if video_mode == "Both models side-by-side" else width

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, output_fps, (output_width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = max_input_frames

    target_input_frames = min(total_frames, max_input_frames)

    read_count = 0
    inferred_count = 0
    written_count = 0

    progress = st.progress(0)
    status_box = st.empty()

    while read_count < target_input_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if read_count % frame_stride == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            if video_mode == "ResNet only":
                result = process_image(
                    model=resnet_model,
                    model_name=RESNET_MODEL_NAME,
                    display_name="ResNet",
                    image_rgb=frame_rgb,
                    device=device,
                    alpha=alpha,
                    infer_height=infer_height,
                    infer_width=infer_width,
                    roi_top_y=roi_top_y,
                    roi_bottom_width=roi_bottom_width,
                    roi_top_width=roi_top_width,
                )
                final_rgb = add_panel_title(result["final_image"], RESNET_MODEL_NAME)

            elif video_mode == "SegFormer only":
                result = process_image(
                    model=segformer_model,
                    model_name=VIT_MODEL_NAME,
                    display_name="SegFormer",
                    image_rgb=frame_rgb,
                    device=device,
                    alpha=alpha,
                    infer_height=infer_height,
                    infer_width=infer_width,
                    roi_top_y=roi_top_y,
                    roi_bottom_width=roi_bottom_width,
                    roi_top_width=roi_top_width,
                )
                final_rgb = add_panel_title(result["final_image"], VIT_MODEL_NAME)

            else:
                _, _, final_rgb = process_two_models_image(
                    resnet_model=resnet_model,
                    segformer_model=segformer_model,
                    image_rgb=frame_rgb,
                    device=device,
                    alpha=alpha,
                    infer_height=infer_height,
                    infer_width=infer_width,
                    roi_top_y=roi_top_y,
                    roi_bottom_width=roi_bottom_width,
                    roi_top_width=roi_top_width,
                )

            final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
            writer.write(final_bgr)

            inferred_count += 1
            written_count += 1

        read_count += 1

        if read_count % 10 == 0 or read_count == target_input_frames:
            progress.progress(min(read_count / max(target_input_frames, 1), 1.0))
            status_box.write(
                f"Read {read_count}/{target_input_frames} frames | "
                f"Inferred {inferred_count} frames"
            )

    cap.release()
    writer.release()

    progress.empty()
    status_box.empty()

    return {
        "input_frames_read": read_count,
        "frames_inferred": inferred_count,
        "frames_written": written_count,
        "input_fps": input_fps,
        "output_fps": output_fps,
        "output_width": output_width,
        "height": height,
        "frame_stride": frame_stride,
        "video_mode": video_mode,
    }


# =========================================================
# Streamlit UI
# =========================================================

st.set_page_config(
    page_title="Traffic Collision Warning - Two Semantic Segmentation Models",
    layout="wide",
)

st.title("Traffic Collision Warning using Semantic Segmentation")
st.caption(f"Config loaded from: {CONFIG_PATH}")

device = get_device()
st.sidebar.write(f"Device: `{device}`")

resnet_path = st.sidebar.text_input("ResNet checkpoint path", str(DEFAULT_RESNET_PATH))
vit_path = st.sidebar.text_input("SegFormer checkpoint path", str(DEFAULT_VIT_PATH))

alpha = st.sidebar.slider(
    "Overlay alpha",
    min_value=0.1,
    max_value=0.9,
    value=APP_DEFAULT_ALPHA,
    step=0.05,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Inference speed")

default_infer_index = 0
if APP_DEFAULT_INFERENCE_SIZE in APP_INFERENCE_SIZE_OPTIONS:
    default_infer_index = APP_INFERENCE_SIZE_OPTIONS.index(APP_DEFAULT_INFERENCE_SIZE)

infer_size = st.sidebar.selectbox(
    "Inference size",
    options=APP_INFERENCE_SIZE_OPTIONS,
    index=default_infer_index,
    help="Smaller size is faster but less accurate. 384 is a good demo default.",
)

infer_height = int(infer_size)
infer_width = int(infer_size)

max_video_frames = st.sidebar.number_input(
    "Max input video frames",
    min_value=10,
    max_value=3000,
    value=APP_MAX_VIDEO_FRAMES,
    step=10,
)

frame_stride = st.sidebar.number_input(
    "Frame stride",
    min_value=1,
    max_value=30,
    value=APP_FRAME_STRIDE,
    step=1,
    help="3 means infer only every 3rd frame. Faster, but output video has fewer frames.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Danger ROI")

roi_top_y = st.sidebar.slider(
    "ROI top Y ratio",
    min_value=0.30,
    max_value=0.85,
    value=ROI_TOP_Y_RATIO,
    step=0.01,
    help="Larger value means the trapezoid is lower and shorter.",
)

roi_bottom_width = st.sidebar.slider(
    "ROI bottom width ratio",
    min_value=0.20,
    max_value=1.00,
    value=ROI_BOTTOM_WIDTH_RATIO,
    step=0.01,
)

roi_top_width = st.sidebar.slider(
    "ROI top width ratio",
    min_value=0.05,
    max_value=0.80,
    value=ROI_TOP_WIDTH_RATIO,
    step=0.01,
)

st.sidebar.info(
    "ROI hình thang chỉ là vùng nguy hiểm trong ảnh, không phải khoảng cách thật. "
    "Muốn đo khoảng cách mét cần camera calibration/depth/stereo/LiDAR."
)

if not Path(resnet_path).exists():
    st.error(f"ResNet checkpoint not found: {resnet_path}")
    st.stop()

if not Path(vit_path).exists():
    st.error(f"SegFormer checkpoint not found: {vit_path}")
    st.stop()

with st.spinner("Loading both models..."):
    resnet_model = load_model(
        model_name=RESNET_MODEL_NAME,
        checkpoint_path=resnet_path,
        device_name=str(device),
    )
    segformer_model = load_model(
        model_name=VIT_MODEL_NAME,
        checkpoint_path=vit_path,
        device_name=str(device),
    )

st.success(f"Loaded both models: {RESNET_MODEL_NAME} and {VIT_MODEL_NAME}")

tab_image, tab_video = st.tabs(["Image Demo - Two Models", "Video Demo"])


with tab_image:
    uploaded_image = st.file_uploader(
        "Upload traffic image",
        type=["jpg", "jpeg", "png"],
        key="image_uploader",
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        image_rgb = np.array(image)

        with st.spinner("Running both models..."):
            resnet_result, segformer_result, comparison = process_two_models_image(
                resnet_model=resnet_model,
                segformer_model=segformer_model,
                image_rgb=image_rgb,
                device=device,
                alpha=alpha,
                infer_height=infer_height,
                infer_width=infer_width,
                roi_top_y=roi_top_y,
                roi_bottom_width=roi_bottom_width,
                roi_top_width=roi_top_width,
            )

        st.subheader("Collision Warning Comparison")

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("ResNet status", resnet_result["status"])
            st.metric("ResNet risk score", f"{resnet_result['risk_score']:.4f}")
        with col_m2:
            st.metric("SegFormer status", segformer_result["status"])
            st.metric("SegFormer risk score", f"{segformer_result['risk_score']:.4f}")

        st.image(comparison, caption="Final comparison: ResNet vs SegFormer", use_container_width=True)

        st.markdown("### Detailed outputs")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(image_rgb, caption="Input Image", use_container_width=True)

        with col2:
            st.image(resnet_result["mask_color"], caption="ResNet Predicted Mask", use_container_width=True)
            st.image(resnet_result["final_image"], caption="ResNet Overlay + ROI + Warning", use_container_width=True)

        with col3:
            st.image(segformer_result["mask_color"], caption="SegFormer Predicted Mask", use_container_width=True)
            st.image(segformer_result["final_image"], caption="SegFormer Overlay + ROI + Warning", use_container_width=True)


with tab_video:
    uploaded_video = st.file_uploader(
        "Upload traffic video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader",
    )

    video_mode = st.radio(
        "Video output mode",
        APP_VIDEO_MODES,
        index=0,
        horizontal=True,
        help="Both models is slow because it runs two segmenters for each processed frame.",
    )

    if uploaded_video is not None:
        st.video(uploaded_video)

        st.warning(
            "Video inference is slow, especially when comparing both models. "
            "For a quick demo, use inference size 256/384 and frame stride 3-5."
        )

        run_video = st.button("Process video")

        if run_video:
            suffix = Path(uploaded_video.name).suffix.lower()
            if suffix == "":
                suffix = ".mp4"

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_input:
                tmp_input.write(uploaded_video.read())
                input_video_path = tmp_input.name

            tmp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            output_video_path = tmp_output.name
            tmp_output.close()

            try:
                with st.spinner("Processing video..."):
                    info = process_video_compare(
                        resnet_model=resnet_model,
                        segformer_model=segformer_model,
                        input_video_path=input_video_path,
                        output_video_path=output_video_path,
                        device=device,
                        alpha=alpha,
                        max_input_frames=int(max_video_frames),
                        frame_stride=int(frame_stride),
                        infer_height=infer_height,
                        infer_width=infer_width,
                        roi_top_y=roi_top_y,
                        roi_bottom_width=roi_bottom_width,
                        roi_top_width=roi_top_width,
                        video_mode=video_mode,
                    )

                st.success("Video processed successfully.")
                st.write(info)

                st.video(output_video_path)

                with open(output_video_path, "rb") as file:
                    st.download_button(
                        label="Download processed video",
                        data=file,
                        file_name="processed_two_models_collision_warning.mp4",
                        mime="video/mp4",
                    )

            finally:
                if os.path.exists(input_video_path):
                    os.remove(input_video_path)
