# app/inference.py
import torch
from video_diffusion_pytorch import GaussianDiffusion
from model_architecture import UNet3D_Full     

# removed dot before model_architecture

def load_model(ckpt_path, device):
    model = UNet3D_Full(6, 6, 64, 128).to(device)
    diffusion = GaussianDiffusion(
        denoise_fn=model, image_size=128, num_frames=6, timesteps=1000
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    new_state = {}
    for k, v in ckpt['model_state_dict'].items():
        if k.startswith("model."):
            new_k = k.replace("model.", "denoise_fn.")
            new_state[new_k] = v
        else:
            new_state[k] = v

    diffusion.load_state_dict(new_state, strict=False)
    diffusion.eval()
    diffusion.denoise_fn.eval()
    return diffusion

def predict_future_frames(diffusion, input_tensor, device):
    diffusion.eval()
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        return diffusion.sample(cond=input_tensor, batch_size=1).cpu()
