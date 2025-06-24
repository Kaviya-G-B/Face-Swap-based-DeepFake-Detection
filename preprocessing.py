import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2

# ✅ Define dataset parameters
dataset_path = r"D:\frames"  # Path where frames are stored
sequence_length = 10  # Number of frames per sequence
batch_size = 8  # ✅ Matches train.py
img_size = (128, 128)  # ✅ Matches train.py

# ✅ Define the DeepfakeDataset class
class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.video_folders = []
        self.labels = []

        print(f"🔍 Scanning dataset directory: {root_dir}")

        # ✅ Scan all subfolders inside `frames`
        for subfolder in os.listdir(root_dir):
            subfolder_path = os.path.join(root_dir, subfolder)

            if os.path.isdir(subfolder_path):
                # ✅ Assign label based on folder name
                label = 1 if "fake" in subfolder.lower() else 0

                self.video_folders.append(subfolder_path)
                self.labels.append(label)

        # ✅ Debugging: Print dataset size
        print(f"✅ Total video sequences collected: {len(self.video_folders)}")

    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        video_folder = self.video_folders[idx]
        label = self.labels[idx]
        frames = []
        frame_files = sorted(os.listdir(video_folder))[:sequence_length]

        for frame_file in frame_files:
            frame_path = os.path.join(video_folder, frame_file)
            image = cv2.imread(frame_path)

            if image is None:
                print(f"🚨 Skipping corrupted image: {frame_path}")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

            if self.transform:
                image = self.transform(image)

            frames.append(image)

        # ✅ If no valid frames found, return zero tensor
        if len(frames) == 0:
            return torch.zeros((sequence_length, 3, *img_size)), torch.tensor(label, dtype=torch.float32)

        # ✅ Pad sequences if they have fewer than `sequence_length` frames
        while len(frames) < sequence_length:
            frames.append(torch.zeros_like(frames[0]))  

        video_tensor = torch.stack(frames)  # ✅ Shape: [sequence_length, C, H, W]
        return video_tensor, torch.tensor(label, dtype=torch.float32)

# ✅ Define preprocessing transformations
transform = transforms.Compose([
    transforms.Resize(img_size),  # ✅ Matches train.py
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ✅ Main function for preprocessing
def main():
    dataset = DeepfakeDataset(root_dir=dataset_path, transform=transform)

    # ✅ Handle case where no sequences are found
    if len(dataset) == 0:
        print("🚨 No video sequences found! Please check your dataset structure.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    print(f"✅ Total video sequences loaded: {len(dataset)}")

    # ✅ Iterate through one batch to check correctness
    for i, (videos, labels) in enumerate(dataloader):
        print(f"📦 Batch {i+1}: Shape: {videos.shape}, Labels: {labels.shape}")
        print(f"🔹 First 5 labels: {labels[:5].tolist()}")
        break  # Stop after first batch for debugging

if __name__ == "__main__":
    main()
