import os
import re
import time
import gc
import shutil
import subprocess
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

APP_VIDEO_CODEC = str(getattr(cfg, "APP_VIDEO_CODEC", "avc1"))
APP_VIDEO_FALLBACK_CODEC = str(getattr(cfg, "APP_VIDEO_FALLBACK_CODEC", "mp4v"))
FFMPEG_PATH = str(getattr(cfg, "FFMPEG_PATH", "ffmpeg"))
APP_ALLOW_MP4V_FALLBACK = bool(getattr(cfg, "APP_ALLOW_MP4V_FALLBACK", True))


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


def get_warning_color_bgr(status: str) -> tuple[int, int, int]:
    """Return OpenCV BGR color for the warning trapezoid."""
    status = str(status).upper()

    if status == "SAFE":
        return (0, 200, 0)      # green
    if status == "WARNING":
        return (0, 220, 255)    # yellow
    return (0, 0, 255)          # red


def draw_roi_on_image(
    image_rgb: np.ndarray,
    roi: np.ndarray,
    status: str = "SAFE",
    risk_score: float | None = None,
    fill_alpha: float = 0.28,
) -> np.ndarray:
    """
    Draw the warning trapezoid on an RGB image.

    SAFE    -> green
    WARNING -> yellow
    DANGER  -> red
    """
    output_bgr = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    roi_mask = roi.astype(np.uint8)

    color = get_warning_color_bgr(status)

    # Fill the ROI with a transparent warning color.
    colored_layer = output_bgr.copy()
    colored_layer[roi_mask == 1] = color
    output_bgr = cv2.addWeighted(
        colored_layer,
        fill_alpha,
        output_bgr,
        1.0 - fill_alpha,
        0,
    )

    # Draw a thick border around the trapezoid.
    contours, _ = cv2.findContours(
        roi_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(output_bgr, contours, -1, color, 4)

    # Put a small label inside the trapezoid.
    ys, xs = np.where(roi_mask == 1)
    if len(xs) > 0 and len(ys) > 0:
        label = str(status).upper()
        if risk_score is not None:
            label = f"{label} | risk={risk_score:.3f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.75
        thickness = 2
        text_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
        text_w, text_h = text_size

        image_h, image_w = output_bgr.shape[:2]
        center_x = int(np.mean(xs))
        top_y = int(np.min(ys))

        text_x = int(np.clip(center_x - text_w // 2, 10, max(10, image_w - text_w - 10)))
        text_y = int(np.clip(top_y + text_h + 20, text_h + 10, image_h - 10))

        cv2.rectangle(
            output_bgr,
            (text_x - 8, text_y - text_h - 8),
            (text_x + text_w + 8, text_y + baseline + 8),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            output_bgr,
            label,
            (text_x, text_y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

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
    overlay_roi = draw_roi_on_image(overlay, roi, status=status, risk_score=risk_score)
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

def safe_remove_file(file_path: str, retries: int = 8, delay: float = 0.25):
    """Remove temp files safely on Windows without crashing Streamlit."""
    if not file_path:
        return

    for _ in range(retries):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            return
        except PermissionError:
            gc.collect()
            time.sleep(delay)
        except FileNotFoundError:
            return

    # Do not crash only because a temporary video is still locked by Windows.
    try:
        if os.path.exists(file_path):
            st.warning(f"Temporary file is still locked and was not deleted: {file_path}")
    except Exception:
        pass


def find_ffmpeg_executable():
    """
    Find ffmpeg without requiring manual Windows PATH setup.

    Search order:
    1. FFMPEG_PATH from config.py
    2. ffmpeg available in PATH
    3. imageio-ffmpeg package installed by pip
    """
    candidates = []

    if FFMPEG_PATH:
        candidates.append(FFMPEG_PATH)

    candidates.append("ffmpeg")

    for candidate in candidates:
        if not candidate:
            continue

        candidate_path = Path(candidate)
        if candidate_path.exists():
            return str(candidate_path)

        found = shutil.which(candidate)
        if found:
            return found

    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg_exe and Path(ffmpeg_exe).exists():
            return ffmpeg_exe
    except Exception:
        pass

    return None


def open_video_writer(video_path: str, codec: str, fps: float, frame_size: tuple[int, int]):
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
    if writer.isOpened():
        return writer
    writer.release()
    return None


def reencode_to_h264(input_path: str, output_path: str, ffmpeg_exe: str):
    """Re-encode a video to browser-compatible H.264 MP4."""
    command = [
        ffmpeg_exe,
        "-y",
        "-i",
        input_path,
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output_path,
    ]

    subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def create_browser_compatible_video_writer(
    output_video_path: str,
    output_fps: float,
    frame_size: tuple[int, int],
):
    """
    Prefer direct H.264 writing. If OpenCV cannot write H.264 on Windows,
    write a temporary mp4v video and re-encode with ffmpeg/imageio-ffmpeg.
    """
    writer = open_video_writer(
        video_path=output_video_path,
        codec=APP_VIDEO_CODEC,
        fps=output_fps,
        frame_size=frame_size,
    )

    if writer is not None:
        return {
            "writer": writer,
            "write_path": output_video_path,
            "needs_reencode": False,
            "browser_compatible": True,
            "codec_used": APP_VIDEO_CODEC,
            "ffmpeg_exe": None,
            "warning": None,
        }

    raw_path = str(Path(output_video_path).with_name(Path(output_video_path).stem + "_raw.mp4"))
    fallback_writer = open_video_writer(
        video_path=raw_path,
        codec=APP_VIDEO_FALLBACK_CODEC,
        fps=output_fps,
        frame_size=frame_size,
    )

    if fallback_writer is None:
        raise RuntimeError(
            f"Cannot create video writer with codec {APP_VIDEO_CODEC} or "
            f"fallback codec {APP_VIDEO_FALLBACK_CODEC}."
        )

    ffmpeg_exe = find_ffmpeg_executable()
    if ffmpeg_exe is not None:
        return {
            "writer": fallback_writer,
            "write_path": raw_path,
            "needs_reencode": True,
            "browser_compatible": True,
            "codec_used": APP_VIDEO_FALLBACK_CODEC,
            "ffmpeg_exe": ffmpeg_exe,
            "warning": None,
        }

    if not APP_ALLOW_MP4V_FALLBACK:
        fallback_writer.release()
        safe_remove_file(raw_path)
        raise RuntimeError(
            "Codec avc1/H.264 is not available in this OpenCV build, and ffmpeg "
            "was not found. Install imageio-ffmpeg with: pip install imageio-ffmpeg"
        )

    return {
        "writer": fallback_writer,
        "write_path": raw_path,
        "needs_reencode": False,
        "browser_compatible": False,
        "codec_used": APP_VIDEO_FALLBACK_CODEC,
        "ffmpeg_exe": None,
        "warning": (
            "H.264 is not available and ffmpeg was not found. The app exported "
            "an mp4v MP4 file. Download may work, but browser preview may not play."
        ),
    }


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
    cap = None
    writer = None
    writer_info = None
    progress = None
    status_box = None

    read_count = 0
    inferred_count = 0
    written_count = 0
    input_fps = 20
    output_fps = 20
    output_width = 0
    height = 0

    try:
        cap = cv2.VideoCapture(input_video_path)

        if not cap.isOpened():
            raise RuntimeError("Cannot open uploaded video.")

        input_fps = cap.get(cv2.CAP_PROP_FPS)
        if input_fps <= 0 or np.isnan(input_fps):
            input_fps = 20

        output_fps = max(input_fps / max(frame_stride, 1), 1)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width <= 0 or height <= 0:
            raise RuntimeError("Cannot read video width/height from uploaded video.")

        output_width = width * 2 if video_mode == "Both models side-by-side" else width
        frame_size = (output_width, height)

        writer_info = create_browser_compatible_video_writer(
            output_video_path=output_video_path,
            output_fps=output_fps,
            frame_size=frame_size,
        )
        writer = writer_info["writer"]

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = max_input_frames

        target_input_frames = min(total_frames, max_input_frames)

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

    finally:
        if cap is not None:
            cap.release()

        if writer is not None:
            writer.release()

        gc.collect()

        if progress is not None:
            progress.empty()

        if status_box is not None:
            status_box.empty()

    browser_compatible = True
    codec_used = None
    warning = None
    reencoded = False
    raw_write_path = None

    if writer_info is not None:
        browser_compatible = bool(writer_info["browser_compatible"])
        codec_used = writer_info["codec_used"]
        warning = writer_info["warning"]
        raw_write_path = writer_info["write_path"]

        if writer_info["needs_reencode"]:
            reencoded_path = str(Path(output_video_path).with_name(Path(output_video_path).stem + "_h264.mp4"))
            reencode_to_h264(
                input_path=writer_info["write_path"],
                output_path=reencoded_path,
                ffmpeg_exe=writer_info["ffmpeg_exe"],
            )
            os.replace(reencoded_path, output_video_path)
            safe_remove_file(writer_info["write_path"])
            reencoded = True

        elif writer_info["write_path"] != output_video_path:
            os.replace(writer_info["write_path"], output_video_path)

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
        "codec_used": codec_used,
        "reencoded_to_h264": reencoded,
        "browser_compatible": browser_compatible,
        "raw_write_path": raw_write_path,
        "warning": warning,
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

                with open(output_video_path, "rb") as file:
                    output_video_bytes = file.read()

                if info.get("warning"):
                    st.warning(info["warning"])

                if info.get("browser_compatible", True):
                    st.video(output_video_bytes)
                else:
                    st.info(
                        "Preview trên trình duyệt có thể không phát vì video không phải H.264. "
                        "Bạn hãy tải file về và mở bằng VLC hoặc trình phát video trên máy."
                    )

                st.download_button(
                    label="Download processed video",
                    data=output_video_bytes,
                    file_name="processed_two_models_collision_warning.mp4",
                    mime="video/mp4",
                )

            finally:
                safe_remove_file(input_video_path)
                safe_remove_file(output_video_path)
