import os
from PIL import Image, ImageFilter
from tqdm import tqdm

# Paths
hr_folder = 'train/HR'
lr_folder = 'train/LR'

# Create LR folder if not exist
os.makedirs(lr_folder, exist_ok=True)

# Parameters
blur_radius = 1.5  # Gaussian blur strength (1.0 to 2.0 is common for 2x simulation)

# Process images
image_files = [f for f in os.listdir(hr_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
print(f"Processing {len(image_files)} images...")

for img_name in tqdm(image_files):
    img_path = os.path.join(hr_folder, img_name)
    img = Image.open(img_path).convert("RGB")

    # Apply Gaussian blur
    blurred_img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Save to LR folder
    save_path = os.path.join(lr_folder, img_name)
    blurred_img.save(save_path)

print("✅ All images processed and saved to LR folder.")
