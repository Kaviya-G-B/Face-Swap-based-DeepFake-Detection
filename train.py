import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import cv2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from facenet_pytorch import MTCNN
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image  # ✅ Import PIL
import torch
torch.cuda.empty_cache()
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.enabled = True

# ✅ Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Define Paths
frames_path = r"D:\frames"
img_size = (128, 128)
sequence_length = 10  

# ✅ Face Detection using MTCNN
mtcnn = MTCNN(margin=20, keep_all=False, post_process=False, device=device)

# ✅ Define Deepfake Dataset with Data Augmentation
class DeepfakeDataset(Dataset):
    def __init__(self, file_paths, labels, augment=False):  # ✅ Fixed Argument
        self.file_paths = file_paths
        self.labels = labels
        base_transforms = [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]

        augment_transforms = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ]

        self.transform = transforms.Compose(augment_transforms + base_transforms if augment else base_transforms)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        video_folder = self.file_paths[idx]
        label = self.labels[idx]
        frames = []
        frame_files = sorted(os.listdir(video_folder))[:sequence_length]

        for frame_file in frame_files:
            frame_path = os.path.join(video_folder, frame_file)
            image = cv2.imread(frame_path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)  # ✅ Convert to PIL Image
            image = self.transform(image)
            frames.append(image)

        if len(frames) == 0:
            return torch.zeros((sequence_length, 3, *img_size)), torch.tensor(label, dtype=torch.float32)

        while len(frames) < sequence_length:
            frames.append(torch.zeros_like(frames[0]))  # ✅ Correct padding

        video_tensor = torch.stack(frames)
        return video_tensor, torch.tensor(label, dtype=torch.float32)

# ✅ Load Dataset
all_folders = [os.path.join(frames_path, folder) for folder in os.listdir(frames_path)]
labels = [1 if "fake" in folder.lower() else 0 for folder in all_folders]
train_paths, test_paths, train_labels, test_labels = train_test_split(all_folders, labels, test_size=0.2, random_state=42)

train_dataset = DeepfakeDataset(train_paths, train_labels, augment=True)
test_dataset = DeepfakeDataset(test_paths, test_labels, augment=False)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

# ✅ Define Model with Fine-Tuned EfficientNet-B3
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        self.efficientnet = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        self.efficientnet.classifier = nn.Identity()

        # ✅ Fine-tune last few layers
        for param in self.efficientnet.features[:-2].parameters():
            param.requires_grad = False

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(1536, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.lstm = nn.LSTM(128, 256, batch_first=True, num_layers=2, dropout=0.4)
        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.shape
        x = x.view(batch_size * seq_len, C, H, W)
        x = self.efficientnet(x)  
        x = x.view(batch_size * seq_len, 1536, 1, 1)
        x = self.deconv(x)
        x = x.view(batch_size, seq_len, -1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.fc(x)

# ✅ Initialize Model, Optimizer, Loss, and Scheduler
model = DeepfakeDetector().to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=5e-5)
criterion = nn.BCEWithLogitsLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

# ✅ Training Function
def train_model(model, train_loader, optimizer, criterion, scheduler, epochs=50):
    model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)

        # ✅ Scheduler should be called with validation loss, so we skip it for now
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}, Accuracy: {correct / total:.4f}")
        
if __name__ == "__main__":
    train_model(model, train_loader, optimizer, criterion, scheduler, epochs=50)
    
    # ✅ Save Model
    torch.save(model.state_dict(), "deepfake_detected.pth")

