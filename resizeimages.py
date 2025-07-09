import os
from PIL import Image

# Input and output paths
input_folder = 'train/teacher_outputs'
output_folder = 'train/teacher_resized'
os.makedirs(output_folder, exist_ok=True)

# Resize settings
target_size = (256, 256)

# Get and sort image filenames
all_images = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')])

# Resize and save
for fname in all_images:
    img_path = os.path.join(input_folder, fname)
    img = Image.open(img_path)
    resized_img = img.resize(target_size, Image.BICUBIC)
    resized_img.save(os.path.join(output_folder, fname))

print("✅ Resized and saved 100 images to", output_folder)
