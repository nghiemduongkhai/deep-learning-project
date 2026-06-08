
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
from transformers import SegformerForSemanticSegmentation


# =========================
# Config
# =========================

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

DANGER_CLASS_IDS = [
    11,  # person
    12,  # rider
    13,  # car
    14,  # truck
    15,  # bus
    17,  # motorcycle
    18,  # bicycle
]

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# =========================
# Paths
# =========================

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent if APP_DIR.name.lower() == "app" else APP_DIR
MODEL_DIR = PROJECT_DIR / "Model"

DEFAULT_RESNET_PATH = MODEL_DIR / "deeplabv3_resnet50_best.pth"
DEFAULT_VIT_PATH = MODEL_DIR / "segformer_b0_best.pth"


# =========================
# Utility functions
# =========================

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def decode_segmentation_mask(mask: np.ndarray) -> np.ndarray:
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for class_id in range(NUM_CLASSES):
        color_mask[mask == class_id] = CITYSCAPES_COLORS[class_id]

    return color_mask


def preprocess_image(image_rgb: np.ndarray):
    original_h, original_w = image_rgb.shape[:2]

    resized = cv2.resize(
        image_rgb,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )

    image_float = resized.astype(np.float32) / 255.0
    image_float = (image_float - IMAGENET_MEAN) / IMAGENET_STD

    tensor = torch.from_numpy(image_float).permute(2, 0, 1).float().unsqueeze(0)

    return tensor, (original_h, original_w)


def create_danger_roi(height: int, width: int) -> np.ndarray:
    roi = np.zeros((height, width), dtype=np.uint8)

    polygon = np.array([[
        (int(0.25 * width), height),
        (int(0.75 * width), height),
        (int(0.60 * width), int(0.55 * height)),
        (int(0.40 * width), int(0.55 * height)),
    ]], dtype=np.int32)

    cv2.fillPoly(roi, polygon, 1)

    return roi


def compute_collision_risk(pred_mask: np.ndarray):
    h, w = pred_mask.shape

    roi = create_danger_roi(h, w)

    danger_mask = np.isin(pred_mask, DANGER_CLASS_IDS).astype(np.uint8)
    overlap = danger_mask * roi

    roi_area = roi.sum() + 1e-6
    overlap_ratio = float(overlap.sum() / roi_area)

    if overlap_ratio < 0.03:
        status = "SAFE"
    elif overlap_ratio < 0.10:
        status = "WARNING"
    else:
        status = "DANGER"

    return status, overlap_ratio, roi


def draw_roi_on_image(image_rgb: np.ndarray, roi: np.ndarray) -> np.ndarray:
    output = image_rgb.copy()

    contours, _ = cv2.findContours(
        roi.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.drawContours(output_bgr, contours, -1, (0, 255, 255), 3)

    return cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)


def overlay_mask(image_rgb: np.ndarray, pred_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color_mask = decode_segmentation_mask(pred_mask)

    if color_mask.shape[:2] != image_rgb.shape[:2]:
        color_mask = cv2.resize(
            color_mask,
            (image_rgb.shape[1], image_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    overlay = cv2.addWeighted(image_rgb, 1 - alpha, color_mask, alpha, 0)
    return overlay


def add_status_text(image_rgb: np.ndarray, status: str, risk_score: float) -> np.ndarray:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if status == "SAFE":
        color = (0, 200, 0)
    elif status == "WARNING":
        color = (0, 200, 255)
    else:
        color = (0, 0, 255)

    text = f"{status} | risk={risk_score:.3f}"

    cv2.rectangle(image_bgr, (15, 15), (430, 70), (0, 0, 0), -1)
    cv2.putText(
        image_bgr,
        text,
        (25, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2,
        cv2.LINE_AA
    )

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


# =========================
# Model builders
# =========================

def build_deeplabv3_resnet50(num_classes: int):
    model = deeplabv3_resnet50(
        weights=None,
        aux_loss=True,
    )

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
    return model


@st.cache_resource
def load_model(model_name: str, checkpoint_path: str, device_name: str):
    device = torch.device(device_name)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    num_classes = checkpoint.get("num_classes", NUM_CLASSES)

    if model_name == "DeepLabV3-ResNet50":
        model = build_deeplabv3_resnet50(num_classes)
        model.load_state_dict(checkpoint["model"])
    else:
        model = build_segformer_b0(num_classes)
        model.load_state_dict(checkpoint["model"])

    model = model.to(device)
    model.eval()

    return model


def predict_mask(model, model_name: str, image_rgb: np.ndarray, device):
    input_tensor, original_size = preprocess_image(image_rgb)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        if model_name == "DeepLabV3-ResNet50":
            outputs = model(input_tensor)
            logits = outputs["out"]
        else:
            outputs = model(pixel_values=input_tensor)
            logits = outputs.logits

        logits = F.interpolate(
            logits,
            size=(IMAGE_HEIGHT, IMAGE_WIDTH),
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


def process_image(model, model_name: str, image_rgb: np.ndarray, device, alpha: float = 0.45):
    pred_mask = predict_mask(model, model_name, image_rgb, device)

    status, risk_score, roi = compute_collision_risk(pred_mask)

    mask_color = decode_segmentation_mask(pred_mask)
    overlay = overlay_mask(image_rgb, pred_mask, alpha=alpha)
    overlay_roi = draw_roi_on_image(overlay, roi)
    final_image = add_status_text(overlay_roi, status, risk_score)

    return {
        "pred_mask": pred_mask,
        "mask_color": mask_color,
        "overlay": overlay,
        "final_image": final_image,
        "status": status,
        "risk_score": risk_score,
        "roi": roi,
    }


def process_video(
    model,
    model_name: str,
    input_video_path: str,
    output_video_path: str,
    device,
    alpha: float = 0.45,
    max_frames: int = 300,
    process_every_n_frames: int = 1,
):
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        raise RuntimeError("Cannot open uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 20

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    processed_count = 0
    last_final_rgb = None

    progress = st.progress(0)
    status_box = st.empty()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = max_frames

    target_frames = min(total_frames, max_frames)

    while frame_count < max_frames:
        ret, frame_bgr = cap.read()

        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if frame_count % process_every_n_frames == 0 or last_final_rgb is None:
            result = process_image(
                model=model,
                model_name=model_name,
                image_rgb=frame_rgb,
                device=device,
                alpha=alpha,
            )
            final_rgb = result["final_image"]
            last_final_rgb = final_rgb
            processed_count += 1
        else:
            final_rgb = last_final_rgb

        final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
        writer.write(final_bgr)

        frame_count += 1

        progress_value = min(frame_count / max(target_frames, 1), 1.0)
        progress.progress(progress_value)
        status_box.write(f"Processing frame {frame_count}/{target_frames}")

        if frame_count >= target_frames:
            break

    cap.release()
    writer.release()

    progress.empty()
    status_box.empty()

    return {
        "frames_written": frame_count,
        "frames_inferred": processed_count,
        "fps": fps,
        "width": width,
        "height": height,
    }


# =========================
# Streamlit UI
# =========================

st.set_page_config(
    page_title="Traffic Collision Warning - Semantic Segmentation",
    layout="wide",
)

st.title("Traffic Collision Warning using Semantic Segmentation")

st.write(
    "Demo semantic segmentation cho cảnh báo va chạm giao thông. "
    "Hỗ trợ ảnh và video."
)

device = get_device()
st.sidebar.write(f"Device: `{device}`")

model_name = st.sidebar.selectbox(
    "Choose model",
    ["DeepLabV3-ResNet50", "SegFormer-B0"],
)

if model_name == "DeepLabV3-ResNet50":
    checkpoint_path = st.sidebar.text_input(
        "Checkpoint path",
        str(DEFAULT_RESNET_PATH),
    )
else:
    checkpoint_path = st.sidebar.text_input(
        "Checkpoint path",
        str(DEFAULT_VIT_PATH),
    )

alpha = st.sidebar.slider(
    "Overlay alpha",
    min_value=0.1,
    max_value=0.9,
    value=0.45,
    step=0.05,
)

st.sidebar.markdown("---")
max_video_frames = st.sidebar.number_input(
    "Max video frames to process",
    min_value=10,
    max_value=3000,
    value=300,
    step=10,
)

process_every_n_frames = st.sidebar.number_input(
    "Process every N frames",
    min_value=1,
    max_value=30,
    value=1,
    step=1,
)

if not Path(checkpoint_path).exists():
    st.error(f"Checkpoint not found: {checkpoint_path}")
    st.stop()

with st.spinner("Loading model..."):
    model = load_model(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        device_name=str(device),
    )

st.success(f"Loaded model: {model_name}")

tab_image, tab_video = st.tabs(["Image Demo", "Video Demo"])


# =========================
# Image tab
# =========================

with tab_image:
    uploaded_image = st.file_uploader(
        "Upload traffic image",
        type=["jpg", "jpeg", "png"],
        key="image_uploader",
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        image_rgb = np.array(image)

        result = process_image(
            model=model,
            model_name=model_name,
            image_rgb=image_rgb,
            device=device,
            alpha=alpha,
        )

        st.subheader("Collision Warning")
        st.metric("Status", result["status"])
        st.metric("Risk score", f"{result['risk_score']:.4f}")

        col1, col2 = st.columns(2)

        with col1:
            st.image(image_rgb, caption="Input Image", use_container_width=True)
            st.image(result["mask_color"], caption="Predicted Segmentation Mask", use_container_width=True)

        with col2:
            st.image(result["overlay"], caption="Segmentation Overlay", use_container_width=True)
            st.image(result["final_image"], caption="Overlay + Danger ROI + Warning", use_container_width=True)


# =========================
# Video tab
# =========================

with tab_video:
    uploaded_video = st.file_uploader(
        "Upload traffic video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader",
    )

    if uploaded_video is not None:
        st.video(uploaded_video)

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
                    info = process_video(
                        model=model,
                        model_name=model_name,
                        input_video_path=input_video_path,
                        output_video_path=output_video_path,
                        device=device,
                        alpha=alpha,
                        max_frames=int(max_video_frames),
                        process_every_n_frames=int(process_every_n_frames),
                    )

                st.success("Video processed successfully.")
                st.write(info)

                st.video(output_video_path)

                with open(output_video_path, "rb") as f:
                    st.download_button(
                        label="Download processed video",
                        data=f,
                        file_name="processed_collision_warning.mp4",
                        mime="video/mp4",
                    )

            finally:
                if os.path.exists(input_video_path):
                    os.remove(input_video_path)
