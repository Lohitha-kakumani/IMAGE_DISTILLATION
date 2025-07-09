from PIL import Image, ImageFilter
import os

def generate_LR_images(hr_dir, lr_dir, method="bicubic"):
    os.makedirs(lr_dir, exist_ok=True)
    for filename in os.listdir(hr_dir):
        hr_path = os.path.join(hr_dir, filename)
        lr_path = os.path.join(lr_dir, filename)

        img = Image.open(hr_path).convert("RGB")
        if method == "bicubic":
            img = img.resize(img.size, resample=Image.BICUBIC)
        elif method == "bilinear":
            img = img.resize(img.size, resample=Image.BILINEAR)

        img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
        img.save(lr_path)

generate_LR_images("dataset/train/HR", "dataset/train/LR", method="bicubic")
