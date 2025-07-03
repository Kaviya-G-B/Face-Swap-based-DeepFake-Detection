# Face-Swap-based-DeepFake-Detection
📂 Dataset
This project uses the FaceForensics++ (FF++) dataset for training and evaluation of face-swap-based deepfake detection models.

🔗 Download
You can download the dataset from the official website:

👉 https://www.kaggle.com/datasets/hungle3401/faceforensics

🗂️ Directory Structure (Expected)
Place the videos in this structure inside your project root:

FF++/
├── real/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
├── fake/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...

🧹 Preprocessing
After downloading, run:
python preprocessing.py

This will extract frames, perform face alignment, and prepare the data for training and evaluation.

