import os
import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm  
from PIL import Image  
import cv2

# ✅ Import dataset and model *without triggering training*
from train import DeepfakeDataset, DeepfakeDetector  

# ✅ Clear GPU Cache Before Running
torch.cuda.empty_cache()

# ✅ Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🚀 Using device: {device}")

# ✅ Define Paths
frames_path = r"D:\frames"
img_size = (128, 128)
sequence_length = 10  

# ✅ Data Augmentation Boost
test_transforms = transforms.Compose([
    transforms.Resize(img_size),
    transforms.RandomHorizontalFlip(p=0.5),  # ✅ Flip to generalize better
    transforms.RandomRotation(degrees=10),  # ✅ Small rotation for robustness
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # ✅ Adjust lighting
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ✅ Load Test Dataset
all_folders = [os.path.join(frames_path, folder) for folder in os.listdir(frames_path)]
labels = [1 if "fake" in folder.lower() else 0 for folder in all_folders]

# ✅ Split dataset (Ensure the split matches training)
from sklearn.model_selection import train_test_split
_, test_paths, _, test_labels = train_test_split(all_folders, labels, test_size=0.2, random_state=42)

test_dataset = DeepfakeDataset(test_paths, test_labels, augment=False)


# ✅ Check if test dataset is empty
if len(test_dataset) == 0:
    print("❌ No test data found! Check dataset path or preprocessing.")
    exit()

# ✅ Define DataLoader (set `num_workers=0` on Windows to avoid freezing)
batch_size = 8  
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)  

# ✅ Load Model
model = DeepfakeDetector().to(device)  # ✅ Move model to GPU
try:
    model.load_state_dict(torch.load("deepfake_detect.pth", map_location=device))  # ✅ Load on correct device
    print("✅ Model loaded successfully!\n")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# ✅ Set Model to Evaluation Mode
model.eval()

# ✅ Evaluation Function with Error Analysis
def evaluate_model(model, test_loader, test_paths):
    all_preds, all_labels, all_probs = [], [], []
    misclassified_samples = []  # ✅ Store misclassified videos
    total_batches = len(test_loader)

    with torch.no_grad():
        for i, (X_batch, y_batch) in enumerate(test_loader):
            start_time = time.time()  # ✅ Track batch processing time

            if X_batch.shape[0] == 0:  # ✅ Skip empty batches
                print(f"⚠️ Skipping empty batch {i+1}")
                continue

            # ✅ Move data to GPU before inference
            X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)

            outputs = model(X_batch).squeeze()  # ✅ Run model on GPU
            probs = torch.sigmoid(outputs).detach().cpu().numpy()  # ✅ Detach and move to CPU
            preds = (probs > 0.5).astype(int)

            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(probs)

            # ✅ Identify Misclassified Samples
            for j in range(len(y_batch)):
                if preds[j] != y_batch[j].cpu().numpy():  
                    misclassified_samples.append(test_paths[i * test_loader.batch_size + j])

            # ✅ Debugging Info
            batch_time = time.time() - start_time
            print(f"🟢 Processed batch {i+1}/{total_batches} in {batch_time:.2f} sec")

    return np.array(all_preds), np.array(all_labels), np.array(all_probs), misclassified_samples


# ✅ Run Evaluation
print("\n🚀 Starting Evaluation on GPU...")
preds, labels, probs, misclassified_samples = evaluate_model(model, test_loader, test_paths)

# ✅ Compute Metrics
accuracy = accuracy_score(labels, preds)
precision = precision_score(labels, preds, zero_division=1)
recall = recall_score(labels, preds, zero_division=1)
f1 = f1_score(labels, preds, zero_division=1)

print(f"\n🔥 Model Evaluation Results 🔥")
print(f"✅ Accuracy:  {accuracy:.4f}")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall:    {recall:.4f}")
print(f"✅ F1 Score:  {f1:.4f}")

# ✅ Plot ROC Curve
fpr, tpr, _ = roc_curve(labels, probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()

# ✅ Print Misclassified Samples
print("\n⚠️ Misclassified Samples ⚠️")
for sample in misclassified_samples[:10]:  # Show first 10 misclassified samples
    print(sample)
print(f"\nTotal Misclassified Samples: {len(misclassified_samples)}")

# ✅ Function to Plot Misclassified Samples
def plot_misclassified_samples(misclassified_samples, num_samples=5):
    print("\n📸 Displaying Misclassified Samples...")
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    
    for i, folder in enumerate(misclassified_samples[:num_samples]):
        frame_files = sorted(os.listdir(folder))[:1]  # Load first frame
        if not frame_files:
            continue

        frame_path = os.path.join(folder, frame_files[0])
        image = cv2.imread(frame_path)
        if image is None:
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axes[i].imshow(image)
        axes[i].axis("off")
        axes[i].set_title("Fake" if "fake" in folder.lower() else "Real")

    plt.show()

# ✅ Show misclassified frames
plot_misclassified_samples(misclassified_samples)

print("\n✅ Evaluation Completed Successfully! 🚀")
