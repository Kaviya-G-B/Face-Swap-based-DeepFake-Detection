import torch
import onnxruntime as ort
import numpy as np
from train import DeepfakeDetector  # Import your trained model

# ✅ Load trained model
model = DeepfakeDetector()
model.load_state_dict(torch.load("deepfake_detect.pth", map_location="cpu"))  # Load best model
model.eval()

# ✅ Define correct dummy input (batch, sequence, channels, height, width)
dummy_input = torch.randn(1, 10, 3, 128, 128)  # Ensure correct shape

# ✅ Export to ONNX with fixed settings
torch.onnx.export(
    model,
    dummy_input,
    "deepfake_detector.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size", 1: "seq_len"}, "output": {0: "batch_size"}}
)

print("✅ ONNX model exported successfully!")

# ✅ Load ONNX model for inference
model_path = "deepfake_detector.onnx"
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

def predict_onnx(input_tensor):
    """Run inference using ONNX model."""
    outputs = session.run(None, {"input": input_tensor.numpy()})
    prediction = torch.sigmoid(torch.tensor(outputs[0]))  # Convert logits to probability
    return prediction.numpy()

print("✅ ONNX model loaded and ready for inference!")

# ✅ Check ONNX Input Shape
onnx_input_shape = session.get_inputs()[0].shape
print(f"ONNX Model Input Shape: {onnx_input_shape}")

# ✅ Sanity Check: Compare PyTorch vs ONNX Output
test_input = torch.randn(1, 10, 3, 128, 128)  # Test sample

# PyTorch Prediction
model.eval()
with torch.no_grad():
    pytorch_output = torch.sigmoid(model(test_input)).cpu().numpy()

# ONNX Prediction
onnx_output = predict_onnx(test_input)

print(f"🔹 PyTorch Output: {pytorch_output}")
print(f"🔹 ONNX Output: {onnx_output}")

# ✅ Check if ONNX Output is close to PyTorch Output
if np.allclose(pytorch_output, onnx_output, atol=1e-5):
    print("✅ ONNX model outputs match PyTorch model!")
else:
    print("❌ Warning: ONNX and PyTorch outputs differ! Check model layers.")

