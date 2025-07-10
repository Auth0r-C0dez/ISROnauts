# app/main.py
from flask import Flask, request, jsonify
import torch
import h5py
import numpy as np
from io import BytesIO
# from .inference import load_model, predict_future_frames
# from .utils import normalize_sequence, calculate_metrics

from inference import load_model, predict_future_frames
from utils import normalize_sequence, calculate_metrics


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
diffusion = load_model("cloud_motion_diffusion_with_time_embed_new_loss_final (1).pth", device)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return "✅ Cloud Motion Backend is running. Use POST /predict to get results."


@app.route("/predict", methods=["POST"])
def predict():
    if "files" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    uploaded_files = request.files.getlist("files")

    if len(uploaded_files) != 8:
        return jsonify({"error": "Exactly 8 HDF5 files must be uploaded (6 input + 2 ground truth)"}), 400

    try:
        bands = ['IMG_VIS', 'IMG_WV', 'IMG_TIR1', 'IMG_TIR2', 'IMG_MIR', 'IMG_SWIR']
        input_sequence = []
        target_sequence = []

        # Process first 6 files → input (T1–T6)
        for file in uploaded_files[:6]:
            with h5py.File(BytesIO(file.read()), "r") as f:
                frame = [np.array(f[band]) for band in bands]
                input_sequence.append(np.stack(frame))

        input_tensor = torch.tensor(np.stack(input_sequence), dtype=torch.float32)
        norm_input = normalize_sequence(input_tensor)  # shape: (1, 6, 6, 128, 128)

        # Process last 2 files → ground truth (T7, T8)
        for file in uploaded_files[6:]:
            with h5py.File(BytesIO(file.read()), "r") as f:
                frame = [np.array(f[band]) for band in bands]
                target_sequence.append(np.stack(frame))

        # Shape: (2, 6, 128, 128) → (6, 2, 128, 128) → (1, 6, 2, 128, 128)
        target_tensor = torch.tensor(np.stack(target_sequence), dtype=torch.float32).permute(1, 0, 2, 3).unsqueeze(0)

        with torch.no_grad():
            output = predict_future_frames(diffusion, norm_input, device)  # shape: (1, 6, 2, 128, 128)

        # Calculate metrics
        metrics = calculate_metrics(output, target_tensor)

        return jsonify({
            "status": "success",
            "output_shape": str(output.shape),
            "metrics": metrics
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
