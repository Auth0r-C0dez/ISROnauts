# app/utils.py
import torch
import torch.nn.functional as F

def normalize_sequence(seq):
    """
    Normalize the input tensor to [-1, 1] per band.
    Input shape: (6, 6, 128, 128)
    Output shape: (1, 6, 6, 128, 128)
    """
    assert seq.shape == (6, 6, 128, 128), f"Expected (6, 6, 128, 128), got {seq.shape}"
    x = seq.permute(1, 0, 2, 3).reshape(6, -1)  # (6 bands, 6*128*128)
    mins = x.min(dim=1, keepdim=True)[0]
    maxs = x.max(dim=1, keepdim=True)[0]
    norm = (x - mins) / (maxs - mins + 1e-5)
    norm = norm * 2 - 1
    return norm.view(6, 6, 128, 128).permute(1, 0, 2, 3).unsqueeze(0)  # (1, 6, 6, 128, 128)

def calculate_metrics(pred, target):
    """
    Calculate MAE, PSNR, SSIM for each predicted frame (T7, T8)
    Inputs:
        pred, target: torch tensors of shape (1, 6, 2, 128, 128)
    Returns:
        Dict with metrics for T7 and T8 individually
    """
    assert pred.shape == target.shape, f"Prediction and target shapes must match. Got {pred.shape} vs {target.shape}"

    pred = pred.squeeze(0)     # (6, 2, 128, 128)
    target = target.squeeze(0) # (6, 2, 128, 128)

    def compute_psnr(mse):
        return 10 * torch.log10(1 / (mse + 1e-8))

    def compute_ssim(img1, img2):
        # Simplified SSIM (per-frame, averaged over bands)
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        mu1 = img1.mean(dim=[1, 2], keepdim=True)
        mu2 = img2.mean(dim=[1, 2], keepdim=True)
        sigma1 = ((img1 - mu1) ** 2).mean(dim=[1, 2], keepdim=True)
        sigma2 = ((img2 - mu2) ** 2).mean(dim=[1, 2], keepdim=True)
        sigma12 = ((img1 - mu1) * (img2 - mu2)).mean(dim=[1, 2], keepdim=True)

        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))
        return ssim_map.mean().item()

    results = {}
    for t in range(2):  # T7 = 0, T8 = 1
        pred_t = pred[:, t, :, :]     # (6, 128, 128)
        target_t = target[:, t, :, :] # (6, 128, 128)

        mae = F.l1_loss(pred_t, target_t).item()
        mse = F.mse_loss(pred_t, target_t).item()
        psnr = compute_psnr(mse).item()
        ssim = compute_ssim(pred_t, target_t)

        results[f"T{7 + t}"] = {
            "mae": round(mae, 6),
            "psnr": round(psnr, 3),
            "ssim": round(ssim, 4)
        }

    return results
