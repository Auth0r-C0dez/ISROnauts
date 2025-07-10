import streamlit as st
import torch
import h5py
import numpy as np
import os
import matplotlib.pyplot as plt

from inference import load_model, predict_future_frames
from utils import normalize_sequence, calculate_metrics

st.set_page_config(layout="wide", page_title="Cloud Motion Prediction")
st.title("☁️ Cloud Motion Prediction and Evaluation")

# Load model once and cache it
@st.cache_resource
def load_diffusion_model():
    ckpt_path = os.path.join("checkpoints", "cloud_motion_diffusion_with_time_embed_new_loss_final.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_model(ckpt_path, device), device

diffusion, device = load_diffusion_model()

# File uploader UI
st.subheader("Upload 8 HDF5 files (T1–T6 inputs + T7, T8 ground truth)")

uploaded_files = st.file_uploader(
    "Upload in order: T1, T2, ..., T8",
    type="h5",
    accept_multiple_files=True
)

if uploaded_files and len(uploaded_files) == 8:
    with st.spinner("Reading and processing input files..."):
        bands = ["IMG_VIS", "IMG_WV", "IMG_TIR1", "IMG_TIR2", "IMG_MIR", "IMG_SWIR"]

        # Build input tensor: (1, 6, 6, 128, 128)
        input_sequence = []
        for file in uploaded_files[:6]:
            with h5py.File(file, 'r') as f:
                frame = [f[band][:] for band in bands]
                input_sequence.append(np.stack(frame))
        input_tensor = torch.tensor(np.stack(input_sequence))  # (6,6,128,128)
        norm_input = normalize_sequence(input_tensor)  # (1,6,6,128,128)

        # Build ground truth tensor: (1, 6, 2, 128, 128)
        target_sequence = []
        for file in uploaded_files[6:]:
            with h5py.File(file, 'r') as f:
                frame = [f[band][:] for band in bands]
                target_sequence.append(np.stack(frame))
        target_tensor = torch.tensor(np.stack(target_sequence))  # (2,6,128,128)
        target_tensor = target_tensor.permute(1, 0, 2, 3).unsqueeze(0)

    with st.spinner("Running prediction..."):
        output = predict_future_frames(diffusion, norm_input, device)
        metrics = calculate_metrics(output, target_tensor)

    st.success("✅ Prediction complete!")

    st.subheader("📊 Accuracy Metrics")
    st.json(metrics)

    # Visualization
    def show_frame_grid(pred, real, frame_idx):
        fig, axes = plt.subplots(2, 6, figsize=(15, 5))
        for i in range(6):
            axes[0, i].imshow(pred[i, frame_idx], cmap='gray')
            axes[0, i].set_title(f"Pred Band {i+1}")
            axes[0, i].axis('off')

            axes[1, i].imshow(real[i, frame_idx], cmap='gray')
            axes[1, i].set_title(f"Real Band {i+1}")
            axes[1, i].axis('off')

        st.pyplot(fig)

    pred = output.squeeze(0)         # (6, 2, 128, 128)
    real = target_tensor.squeeze(0)  # (6, 2, 128, 128)

    st.subheader("🖼️ Predicted vs Real: Frame T7")
    show_frame_grid(pred, real, frame_idx=0)

    st.subheader("🖼️ Predicted vs Real: Frame T8")
    show_frame_grid(pred, real, frame_idx=1)

elif uploaded_files:
    st.warning("⚠️ Please upload exactly 8 HDF5 files in the correct order (T1–T8).")
