import torch
from PIL import Image
import torchvision.transforms as transforms
import os
from unet_model import UNet

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load("distilled_unet.pth", map_location=device))
model.eval()

# Image transform
transform = transforms.ToTensor()

# Input and output folders
input_folder = "patches/LR"
output_folder = "patches/HR_Predicted"
os.makedirs(output_folder, exist_ok=True)

# Process each LR image
for filename in os.listdir(input_folder):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        img_path = os.path.join(input_folder, filename)
        lr_img = Image.open(img_path).convert("RGB")
        lr_tensor = transform(lr_img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(lr_tensor).squeeze(0).cpu()
        
        output_img = transforms.ToPILImage()(output.clamp(0, 1))
        output_img.save(os.path.join(output_folder, filename))

print("✅ Predictions saved to:", output_folder)
