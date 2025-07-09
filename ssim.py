import os
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Folders
pred_dir = "patches/HR_Predicted"
gt_dir = "patches/HR"

# List files (assuming same names in both)
pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".png")])[:100]
gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".png")])[:100]

ssim_scores = []

for pred_file, gt_file in zip(pred_files, gt_files):
    pred_img = Image.open(os.path.join(pred_dir, pred_file)).convert("RGB")
    gt_img = Image.open(os.path.join(gt_dir, gt_file)).convert("RGB")

    pred_np = np.array(pred_img)
    gt_np = np.array(gt_img)

    score = ssim(pred_np, gt_np, channel_axis=2, data_range=255)
    ssim_scores.append(score)

# Average SSIM
avg_ssim = sum(ssim_scores) / len(ssim_scores)
print(f"✅ Average SSIM over {len(ssim_scores)} images: {avg_ssim:.4f}")
