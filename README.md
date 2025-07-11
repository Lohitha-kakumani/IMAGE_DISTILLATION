# 📌 Image Sharpening using Knowledge Distillation

## 🚀 Overview
This project enhances image sharpness using a **Knowledge Distillation** framework. A pre-trained **Real-ESRGAN** model (Teacher) generates high-quality sharp outputs, while a lightweight **UNet** model (Student) is trained to mimic the teacher’s performance.  
The final model is designed for fast, real-time image sharpening — ideal for video conferencing under poor network conditions.

---

## 📂 Dataset

We use the **DIV2K** dataset:
- 800 high-resolution (HR) images split into **3200 patches** of size `256x256`
- Corresponding low-resolution (LR) versions are generated using **bicubic downscaling**

📎 **Dataset Location:**  
🔗 [posted in kraggle  – LR, HR] - https://www.kaggle.com/datasets/lohithakakumani/intel-dataset
🔗 [posted in kraggle  – Teacher Outputs] - https://www.kaggle.com/datasets/lohithakakumani/teacher-output
🔗 [Google Drive Link] - [https://www.kaggle.com/datasets/lohithakakumani/teacher-output](https://drive.google.com/drive/folders/1bVjz7PP-XG8DhVRy-eUZQe7eRYwOG01S?usp=sharing)
---

🧠 Teacher Model – RealESRGAN
Used the pre-trained RealESRGAN_x4plus model for high-quality image enhancement
Ran over 3200 Low-Resolution (LR) image patches
Generated sharp and detailed outputs that act as distillation targets for training the student model
📎 Output Location
📁 teacher_outputs/ – All outputs resized to 256x256 to match the input size for student training


👶 Student Model – Lightweight UNet
The student model is a compact UNet architecture trained using:
✅ Pixel-wise L1 Loss (between prediction & HR ground truth)
✅ Distillation Loss (between prediction & teacher output)
📁 Model gets saved as:
distilled_unet.pth

🧪 Testing the Student Model
After training, the student model is evaluated on 400 unseen LR patches.
Output images are saved to:
📁 patches/HR_Predicted/

📈 SSIM Evaluation
To evaluate performance, we compute SSIM (Structural Similarity Index) between predicted and ground truth HR images.
✅ Average SSIM Score: 0.86 (on 100 test images)

📁 Folder Structure
INTEL/
├── dataset/                     # Contains LR, HR images of DIV2K
├── patches/                    # Stores cropped image patches (LR, HR, and predicted outputs of student model)
├── train/                      # Folder contains cropped images of LR,HR, and also teacher_outputs
├── distilled_unet.pth         # ✅ Trained student model (UNet) weights
├── RealESRGAN_x4plus.pth      # 📥 Official RealESRGAN pretrained model file
├── app.py                      # 📥 test script to run the teacher model
├── intel_teacher_model_fixed.pth # ✅ Pretrained RealESRGAN (teacher) model weights
├── testscript.py              # 🧪 Script to test student model and save predictions
├── ssim.py                    # 📏 Calculates SSIM between predictions and ground truth
├── unet_model.py              # 🧠 Lightweight UNet student model architecture
├── cropimage.py               # ✂️ Crops HR images into 256x256 patches
├── lowscale.py                # 🔽 Generates LR images using bicubic downscaling
├── requirements.txt           # 📚 Python dependencies for the project
├── intel.ipynb                # 📓 Jupyter notebook for experimentation or demo
├── INTEL_PRESENTATION.pptx    # 🖼️ Final project presentation file
├── Intel_report.pdf           # 📄 Final project report (includes models, methodology, results)
├── INTEL_VIDEO_EXPLANATION.mp4 # 🎥 Screen-recorded explanation of the project


## ⚙️ Environment Setup

> Python ≥ 3.9, CUDA GPU recommended

```bash
conda create -n intel-env-fresh python=3.9
conda activate intel-env-fresh
pip install -r requirements.txt

