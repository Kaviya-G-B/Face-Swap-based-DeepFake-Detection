import os
import cv2
import numpy as np
import torch
import onnxruntime
from PIL import Image
from flask import Flask, Response, request, jsonify, send_from_directory

from flask_cors import CORS
from facenet_pytorch import MTCNN
from torchvision import transforms
import time

os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_DSHOW"] = "1"

app = Flask(__name__, static_folder="react-front--main/dist", static_url_path="")
CORS(app)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = "deepfake_detector.onnx"
ort_session = onnxruntime.InferenceSession(MODEL_PATH)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ Using device: {device}")
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mtcnn = MTCNN(keep_all=True, device=device, thresholds=[0.6, 0.7, 0.7])

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

camera = None
streaming = False

def start_camera():
    global camera, streaming
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not camera.isOpened():
            raise RuntimeError("❌ Failed to open camera")

        streaming = True
        print("✅ Camera started")

def stop_camera():
    global camera, streaming
    streaming = False
    if camera is not None:
        camera.release()
        camera = None
        print("✅ Camera stopped")

def predict(face):
    try:
        face_tensor = transform(face).unsqueeze(0).unsqueeze(0).numpy()

        output = ort_session.run(None, {"input": face_tensor})
        score = output[0][0][0]
        label = "Fake" if score > 0.5 else "Real"
        return label, score
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return "Error", 0.0

def detect_and_predict(frame):
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = image.crop((x, y, x + w, y + h))
            label, score = predict(face)
            color = (0, 255, 0) if label == "Real" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} ({score:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame
    except Exception as e:
        print(f"❌ Detection error: {e}")
        return frame
def generate_frames():
    global camera, streaming
    while streaming:
        success, frame = camera.read()
        if not success:
            print("❌ Failed to read frame from camera")
            break

        frame = detect_and_predict(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
@app.route('/start-stream')
def start_stream():
    try:
        start_camera()
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"❌ Stream error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/stop-stream', methods=['POST'])
def stop_stream():
    stop_camera()
    return jsonify({"status": "Camera stopped"}), 200
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def catch_all(path):
    return send_from_directory(app.static_folder, path)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000)
