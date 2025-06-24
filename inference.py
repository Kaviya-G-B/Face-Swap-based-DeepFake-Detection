import os
import cv2
import torch
import numpy as np
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# ✅ Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.init()

# ✅ Load the trained model
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        self.efficientnet = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        self.efficientnet.classifier = nn.Identity()

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
        x = x.view(batch_size * seq_len, C, H, W)  # ✅ Flatten batch & sequence dimensions
    
        x = self.efficientnet.features(x)  # ✅ Extract CNN features
        b, c, h, w = x.shape  # ✅ Dynamically get shape

        x = self.deconv(x)  # ✅ Deconvolution
        x = x.view(batch_size, seq_len, -1)  # ✅ Flatten for LSTM

        x, _ = self.lstm(x)  # ✅ LSTM should process correctly
        x = x[:, -1, :]  # ✅ Take last LSTM output

        return self.fc(x)  # ✅ Final classification


# ✅ Load Model
model = DeepfakeDetector().to(device)
model.load_state_dict(torch.load("deepfake_detect.pth", map_location=device))
model.eval()

# ✅ Define Image Preprocessing
def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    return transform(frame)

# ✅ Extract Frames from Video
def extract_frames(video_path, sequence_length=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(preprocess_frame(frame))
    cap.release()

    if len(frames) < sequence_length:
        frames.extend([torch.zeros_like(frames[0])] * (sequence_length - len(frames)))
    return torch.stack(frames[:sequence_length]).unsqueeze(0).to(device)

# ✅ Grad-CAM for Heatmap Generation
def grad_cam(model, frames):
    model.train()  # ✅ Enable gradients by switching to training mode

    # Ensure model parameters require gradients
    for param in model.parameters():
        param.requires_grad = True

    # ✅ Get EfficientNet features (Use model.efficientnet.features instead of full model)
    features_extractor = model.efficientnet.features
    features_extractor.requires_grad_(True)

    gradients = []
    activations = []

    # ✅ Hook to capture activations & gradients
    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # ✅ Register hooks on the LAST CONVOLUTIONAL LAYER
    target_layer = features_extractor[-1]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    # ✅ Forward pass through the feature extractor
    features = features_extractor(frames.squeeze(0))

    # ✅ Forward pass through the full model
    output = model(frames).squeeze()
    output.backward(retain_graph=True)  # ✅ Compute gradients

    if not gradients:
        raise RuntimeError("❌ Gradients are missing! Check if hooks are registered.")

    # ✅ Extract activation maps & gradients
    activations = activations[0].detach().cpu().numpy()
    gradients = gradients[0].detach().cpu().numpy()

    # ✅ Compute Grad-CAM weights
    weights = np.mean(gradients, axis=(2, 3))

    # ✅ Compute weighted sum of activations
    cam = np.zeros(activations.shape[2:], dtype=np.float32)
    for i, w in enumerate(weights[0]):
        cam += w * activations[0, i]

    # ✅ Normalize the heatmap
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (128, 128))
    cam = (cam - cam.min()) / (cam.max() - cam.min())

    model.eval()  # ✅ Restore evaluation mode
    return cam



# ✅ Overlay Heatmap on Image
def overlay_heatmap(frame, heatmap):
    # ✅ Ensure heatmap has the same size as the original image
    heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))

    # ✅ Convert heatmap to 3 channels if it's grayscale
    if len(heatmap_resized.shape) == 2:  
        heatmap_resized = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(frame, 0.5, heatmap_resized, 0.5, 0)
    return overlay


# ✅ Predict Single Image
def predict_image(image_path):
    image = preprocess_frame(cv2.imread(image_path)).unsqueeze(0).to(device)  # Shape: (1, C, H, W)
    
    image = image.unsqueeze(1)  # ✅ Add sequence dimension: (1, 1, C, H, W)
    
    with torch.no_grad():
        output = model(image).squeeze().item()

    prediction = "FAKE" if output > 0.5 else "REAL"
    confidence = round(output, 4)
    print(f"Prediction: {prediction} (Confidence: {output:.4f})")

    # ✅ Generate heatmap
    heatmap = grad_cam(model, image)
    original = cv2.imread(image_path)
    overlay = overlay_heatmap(original, heatmap)
    heatmap_path = f"heatmap_{os.path.basename(image_path)}"
    cv2.imwrite(heatmap_path, overlay)
    print(f"Heatmap saved as {heatmap_path}")

# ✅ Predict Video
def predict_video(video_path):
    frames = extract_frames(video_path)
    with torch.no_grad():
        output = model(frames).item()

    prediction = "FAKE" if output > 0.5 else "REAL"
    print(f"Video Prediction: {prediction}")

    heatmap = grad_cam(model, frames)
    original_frame = cv2.imread(video_path)[:128, :128]  # Resize for visualization
    overlay = overlay_heatmap(original_frame, heatmap)
    cv2.imwrite("output_heatmap.jpg", overlay)
    print("Heatmap saved as output_heatmap.jpg")

# ✅ Predict Folder of Images
def predict_folder(folder_path):
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    if not image_files:
        print("❌ No valid images found in the folder.")
        return

    results = []
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)
        if image is None:
            continue

        image_tensor = preprocess_frame(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image_tensor).squeeze().item()

        prediction = "FAKE" if output > 0.5 else "REAL"
        results.append(prediction)
        print(f"Frame {image_file}: {prediction} (Confidence: {output:.4f})")

        # ✅ Save heatmap
        heatmap = grad_cam(model, image_tensor)
        overlay = overlay_heatmap(image, heatmap)
        cv2.imwrite(f"heatmap_{image_file}", overlay)

    # ✅ Majority voting for overall video prediction
    final_prediction = "FAKE" if results.count("FAKE") > results.count("REAL") else "REAL"
    print(f"\nOverall Video Prediction (Majority Vote): {final_prediction}")

# ✅ Main function
if __name__ == "__main__":
    input_path = input("Enter a video file, image file, or folder path: ")

    if os.path.isdir(input_path):
        print("\n🔍 Processing Folder of Images...\n")
        predict_folder(input_path)
    elif os.path.isfile(input_path):
        if input_path.lower().endswith(('.mp4', '.avi', '.mov')):
            print("\n🎥 Processing Video File...\n")
            predict_video(input_path)
        elif input_path.lower().endswith(('.jpg', '.png', '.jpeg')):
            print("\n🖼️ Processing Single Image...\n")
            predict_image(input_path)
        else:
            print("❌ Unsupported file format.")
    else:
        print("❌ Invalid path. Please enter a valid file or folder.")
