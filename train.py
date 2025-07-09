import os
import sys
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as ssim
from PIL import Image

from unet_model import UNet
from dataset import ImagePairDataset

# Add paths to import teacher model
sys.path.append(os.getcwd())
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

# Paths
LR_DIR = 'train/LR'
HR_DIR = 'train/HR'
TEACHER_MODEL_PATH = 'realesrgan_x4plus.pth'

# Dataset and DataLoader
dataset = ImagePairDataset(LR_DIR, HR_DIR)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Student model
student = UNet().to(device)
optimizer = optim.Adam(student.parameters(), lr=1e-4)
mse_loss = nn.MSELoss()

# Teacher model (Real-ESRGAN)
teacher_net = RRDBNet(
    num_in_ch=3, num_out_ch=3,
    num_feat=64, num_block=23,
    num_grow_ch=32, scale=4
)
teacher = RealESRGANer(
    scale=4,
    model_path=TEACHER_MODEL_PATH,
    model=teacher_net,
    tile=128,
    tile_pad=10,
    pre_pad=0,
    half=False,
    device=device
)

# SSIM Loss function
def ssim_loss(pred, target):
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    total = 0
    for i in range(pred.shape[0]):
        p = pred[i].transpose(1, 2, 0)
        t = target[i].transpose(1, 2, 0)
        total += (1 - ssim(p, t, win_size=3, channel_axis=2, data_range=1.0))
    return torch.tensor(total / pred.shape[0])

# Training
epochs = 10
for epoch in range(epochs):
    student.train()
    running_loss = 0.0

    for lr_imgs, hr_imgs, filenames in dataloader:
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)

        # Get teacher predictions for each image in batch
        teacher_outputs = []
        for i in range(lr_imgs.size(0)):
            img_np = (lr_imgs[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            enhanced, _ = teacher.enhance(img_np, outscale=1)
            out_tensor = torch.tensor(enhanced).permute(2, 0, 1).float() / 255.0
            teacher_outputs.append(out_tensor)

        teacher_batch = torch.stack(teacher_outputs).to(device)

        # Forward through student
        student_outputs = student(lr_imgs)

        # Loss calculation
        loss_m = mse_loss(student_outputs, teacher_batch)
        loss_s = ssim_loss(student_outputs, teacher_batch)
        total_loss = loss_m + loss_s

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss:.4f}")
