import numpy as np
import scipy.signal
from PIL import Image

# === Preprocessing for AlexNet ===
def preprocess_image(image_hwc):
    print("Original image shape:", image_hwc.shape)
    if image_hwc.ndim != 3 or image_hwc.shape[2] != 3:
        raise ValueError(f"Expected image shape [H, W, 3], got {image_hwc.shape}")

    image_uint8 = (image_hwc * 255).astype(np.uint8)
    image_pil = Image.fromarray(image_uint8)
    image_pil = image_pil.resize((224, 224), Image.BILINEAR)
    image_resized = np.array(image_pil).astype(np.float32) / 255.0
    image_chw = np.transpose(image_resized, (2, 0, 1))

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    image_normalized = (image_chw - mean) / std

    print("Processed image shape:", image_normalized.shape)
    return image_normalized

# === Quantization Utilities ===
def calculate_scale_zero_point(tensor, num_bits=4):
    qmax = (2 ** (num_bits - 1)) - 1
    max_val = np.max(np.abs(tensor))
    scale = max_val / qmax if max_val != 0 else 1.0
    zero_point = 0
    return scale, zero_point

def quantize_tensor(tensor, scale, zero_point, num_bits=4):
    qmin = -(2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1
    q_tensor = np.round(tensor / scale + zero_point)
    return np.clip(q_tensor, qmin, qmax).astype(np.int8)

def dequantize_tensor(q_tensor, scale, zero_point):
    return scale * (q_tensor.astype(np.float32) - zero_point)

# === Convolution Function ===
def conv2d(input, weight, bias=None, stride=4, padding=0):
    out_channels, in_channels, kh, kw = weight.shape
    _, in_h, in_w = input.shape
    padded = np.pad(input, ((0, 0), (padding, padding), (padding, padding)), mode='constant')
    out_h = (in_h + 2 * padding - kh) // stride + 1
    out_w = (in_w + 2 * padding - kw) // stride + 1
    output = np.zeros((out_channels, out_h, out_w), dtype=np.float32)

    for oc in range(out_channels):
        for ic in range(in_channels):
            output[oc] += scipy.signal.correlate2d(
                padded[ic], weight[oc, ic], mode='valid')[::stride, ::stride]
        if bias is not None:
            output[oc] += bias[oc]
    return output

# === INT4 Pipeline + Comparison ===
def int4_convolution_pipeline(image_path, weights_path, bias_path):
    image = np.load(image_path)  # [H, W, C] in [0, 1]
    image_fp32 = preprocess_image(image)

    weights_fp32 = np.load(weights_path)  # [OC, IC, KH, KW]
    bias_fp32 = np.load(bias_path)        # [OC]

    # --- FP32 Convolution ---
    output_fp32 = conv2d(image_fp32, weights_fp32, bias=bias_fp32, stride=4, padding=0)

    # --- Quantization ---
    scale_x, zp_x = calculate_scale_zero_point(image_fp32)
    image_q = quantize_tensor(image_fp32, scale_x, zp_x)

    scale_w, zp_w = calculate_scale_zero_point(weights_fp32)
    weights_q = quantize_tensor(weights_fp32, scale_w, zp_w)

    # --- Bias Quantization ---
    scale_y = scale_x * scale_w
    bias_q = np.round(bias_fp32 / scale_y).astype(np.int32)

    # --- INT4 Convolution ---
    output_q = conv2d(image_q, weights_q, bias=bias_q, stride=4, padding=0)
    output_dequant = dequantize_tensor(output_q, scale_y, 0)

    # --- Error Metrics ---
    mse = np.mean((output_fp32 - output_dequant) ** 2)
    mae = np.mean(np.abs(output_fp32 - output_dequant))
    max_err = np.max(np.abs(output_fp32 - output_dequant))

    print("\n📊 Quantization Error (INT4 vs FP32):")
    print(f"🔸 Mean Squared Error (MSE): {mse:.6f}")
    print(f"🔸 Mean Absolute Error (MAE): {mae:.6f}")
    print(f"🔸 Max Absolute Error: {max_err:.6f}")

    print("\n📅 FP32 Output (Channel 0, 5x5):\n", output_fp32[0, :5, :5])
    print("\n📅 INT4 Dequantized Output (Channel 0, 5x5):\n", output_dequant[0, :5, :5])

    return output_fp32, output_dequant

# === Main ===
if __name__ == "__main__":
    image_path = "/home/admin1/Python/task_mar_7/image_fp32.npy"
    weights_path = "/home/admin1/Downloads/conv1_w_0.npy"
    bias_path = "/home/admin1/Downloads/conv1_b_0 (1).npy"

    output_fp32, output_dequant = int4_convolution_pipeline(image_path, weights_path, bias_path)
