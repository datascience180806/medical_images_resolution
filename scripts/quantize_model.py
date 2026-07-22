import os
import argparse
import glob
import copy
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor
from PIL import Image
from tqdm import tqdm

from model_architecture import Generator, ConvBlock, SeperableConv2d, ResidualBlock, UpsampleBlock
from model_metrics import ssim


def fuse_conv_bn_eval(conv, bn):
    """
    Fuses a Conv2d and BatchNorm2d layer into a single Conv2d layer for evaluation/quantization.
    """
    fused_conv = copy.deepcopy(conv)
    
    w = conv.weight
    mean = bn.running_mean
    var_val = bn.running_var
    eps = bn.eps
    gamma = bn.weight
    beta = bn.bias

    if gamma is None:
        gamma = torch.ones(conv.out_channels, device=w.device)
    if beta is None:
        beta = torch.zeros(conv.out_channels, device=w.device)

    std = torch.sqrt(var_val + eps)
    t = (gamma / std).reshape(-1, 1, 1, 1)
    
    # Adjust for depthwise vs standard convolution
    if conv.groups > 1 and conv.groups == conv.in_channels:
        # Depthwise conv weight shape: (in_channels, 1, K, K)
        t_conv = (gamma / std).reshape(-1, 1, 1, 1)
    else:
        # Standard or pointwise conv weight shape: (out_channels, in_channels, K, K)
        t_conv = t

    fused_conv.weight = nn.Parameter(w * t_conv)

    if conv.bias is not None:
        b = conv.bias
    else:
        b = torch.zeros(conv.out_channels, device=w.device)

    fused_conv.bias = nn.Parameter((b - mean) * (gamma / std) + beta)
    return fused_conv


def fuse_generator_bn(generator):
    """
    Traverses the Generator architecture and fuses all BatchNorm2d layers into preceding Conv2d layers.
    """
    net = copy.deepcopy(generator)
    net.eval()

    def _fuse_conv_block(block):
        if hasattr(block, 'bn') and isinstance(block.bn, nn.BatchNorm2d):
            # Fuse pointwise or depthwise conv
            block.cnn.pointwise = fuse_conv_bn_eval(block.cnn.pointwise, block.bn)
            block.bn = nn.Identity()

    # Fuse in initial convblock if any
    _fuse_conv_block(net.initial)

    # Fuse in residual blocks
    for res_block in net.residual:
        _fuse_conv_block(res_block.block1)
        _fuse_conv_block(res_block.block2)

    # Fuse in intermediate convblock
    _fuse_conv_block(net.convblock)

    return net


class QuantizedConv2d(nn.Module):
    """
    Wrapper layer for Conv2d storing INT8 weights, scales, zero-points, and tracking activation statistics.
    """
    def __init__(self, conv: nn.Conv2d, num_bits=8):
        super(QuantizedConv2d, self).__init__()
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.groups = conv.groups
        self.num_bits = num_bits

        # 1. Quantize weights per-tensor (or per-channel)
        w_fp32 = conv.weight.data.clone()
        w_min = w_fp32.min().item()
        w_max = w_fp32.max().item()

        # Symmetric INT8 quantization for weights: [-127, 127]
        max_abs = max(abs(w_min), abs(w_max))
        if max_abs == 0:
            scale_w = 1.0
        else:
            scale_w = max_abs / 127.0
        
        w_int8 = torch.clamp(torch.round(w_fp32 / scale_w), -127, 127).to(torch.int8)

        self.register_buffer("weight_int8", w_int8)
        self.register_buffer("weight_scale", torch.tensor(scale_w, dtype=torch.float32))
        self.register_buffer("weight_zero_point", torch.tensor(0, dtype=torch.int32))

        if conv.bias is not None:
            self.register_buffer("bias", conv.bias.data.clone())
        else:
            self.bias = None

        # Dequantized weight for PyTorch forward pass simulation
        self.w_dequantized = (w_int8.float() * scale_w)

        # Activation calibration trackers
        self.register_buffer("act_in_min", torch.tensor(float('inf')))
        self.register_buffer("act_in_max", torch.tensor(float('-inf')))
        self.register_buffer("act_out_min", torch.tensor(float('inf')))
        self.register_buffer("act_out_max", torch.tensor(float('-inf')))

        self.register_buffer("act_in_scale", torch.tensor(1.0))
        self.register_buffer("act_in_zero_point", torch.tensor(0, dtype=torch.int32))
        self.register_buffer("act_out_scale", torch.tensor(1.0))
        self.register_buffer("act_out_zero_point", torch.tensor(0, dtype=torch.int32))

        self.calibrating = True

    def update_act_stats(self, x, min_buf, max_buf):
        current_min = x.min().item()
        current_max = x.max().item()
        min_buf.copy_(torch.tensor(min(min_buf.item(), current_min)))
        max_buf.copy_(torch.tensor(max(max_buf.item(), current_max)))

    def finalize_quantization(self):
        self.calibrating = False
        # Calculate Activation Scale & Zero Point (Asymmetric uint8 [0, 255])
        def calc_scale_zp(a_min, a_max):
            a_min = min(0.0, a_min.item())
            a_max = max(0.0, a_max.item())
            if a_max == a_min:
                return 1.0, 0
            scale = (a_max - a_min) / 255.0
            zp = int(round(-a_min / scale))
            zp = max(0, min(255, zp))
            return scale, zp

        s_in, zp_in = calc_scale_zp(self.act_in_min, self.act_in_max)
        s_out, zp_out = calc_scale_zp(self.act_out_min, self.act_out_max)

        self.act_in_scale.copy_(torch.tensor(s_in, dtype=torch.float32))
        self.act_in_zero_point.copy_(torch.tensor(zp_in, dtype=torch.int32))
        self.act_out_scale.copy_(torch.tensor(s_out, dtype=torch.float32))
        self.act_out_zero_point.copy_(torch.tensor(zp_out, dtype=torch.int32))

    def forward(self, x):
        if self.calibrating:
            self.update_act_stats(x, self.act_in_min, self.act_in_max)

        # Simulate quantized conv forward pass using dequantized int8 weights
        out = nn.functional.conv2d(
            x, self.w_dequantized, self.bias,
            stride=self.stride, padding=self.padding, groups=self.groups
        )

        if self.calibrating:
            self.update_act_stats(out, self.act_out_min, self.act_out_max)

        return out


def replace_conv2d_with_quant(module):
    """
    Recursively replaces all nn.Conv2d layers with QuantizedConv2d.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            setattr(module, name, QuantizedConv2d(child))
        else:
            replace_conv2d_with_quant(child)


def quantize_generator(weights_path, calib_dir, output_model_path, upscale_factor=4, device_str='cpu'):
    """
    Main PTQ Quantization workflow.
    """
    device = torch.device(device_str)
    print(f"[INFO] Running Quantization process on device: {device}")

    # 1. Load FP32 Generator
    print(f"[INFO] Loading FP32 Generator weights from: {weights_path}")
    base_net = Generator(upscale_factor=upscale_factor).to(device)
    checkpoint = torch.load(weights_path, map_location=device)
    if "model" in checkpoint:
        base_net.load_state_dict(checkpoint["model"])
    else:
        base_net.load_state_dict(checkpoint)
    base_net.eval()

    # 2. Fuse BatchNorm layers
    print("[INFO] Fusing Conv2d + BatchNorm2d layers...")
    fused_net = fuse_generator_bn(base_net)

    # 3. Replace Conv2d with QuantizedConv2d
    print("[INFO] Converting Conv2d layers to Quantized INT8 Conv layers...")
    replace_conv2d_with_quant(fused_net)
    fused_net.to(device)

    # 4. Calibration pass
    if not os.path.exists(calib_dir):
        raise FileNotFoundError(f"Calibration directory not found: {calib_dir}")

    calib_files = sorted([
        os.path.join(calib_dir, f) for f in os.listdir(calib_dir)
        if f.endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'))
    ])
    print(f"[INFO] Starting PTQ Calibration on {len(calib_files)} calibration images...")

    lr_transform = transforms.Resize((256, 256), interpolation=Image.BICUBIC)

    with torch.no_grad():
        for img_path in tqdm(calib_files, desc="Calibrating PTQ"):
            hr_pil = Image.open(img_path).convert('RGB')
            lr_pil = lr_transform(hr_pil)
            lr_tensor = to_tensor(lr_pil).unsqueeze(0).to(device)

            _ = fused_net(lr_tensor)

    # 5. Finalize activation scales and zero points
    print("[INFO] Finalizing INT8 Quantization scales & zero points...")
    for m in fused_net.modules():
        if isinstance(m, QuantizedConv2d):
            m.finalize_quantization()

    # 6. Save Quantized Model
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    torch.save({"model": fused_net.state_dict(), "quantized": True}, output_model_path)
    print(f"[INFO] Quantized INT8 model successfully saved to: {output_model_path}")

    # Calculate model size reduction
    fp32_size = os.path.getsize(weights_path) / (1024 * 1024)
    int8_size = os.path.getsize(output_model_path) / (1024 * 1024)
    print(f"[SUMMARY] Original FP32 Model Size : {fp32_size:.2f} MB")
    print(f"[SUMMARY] Quantized INT8 Model Size: {int8_size:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize Swift-SRGAN Generator to INT8 for FPGA Zynq")
    parser.add_argument('--weights', type=str, default='./models/netG_4x_epoch5.pth.tar', help='Path to FP32 weights')
    parser.add_argument('--calib_dir', type=str, default='./calibration_images', help='Path to calibration images')
    parser.add_argument('--output', type=str, default='./models/netG_4x_quantized_int8.pth', help='Output quantized model path')
    parser.add_argument('--upscale_factor', type=int, default=4, help='Upscale factor')
    parser.add_argument('--device', type=str, default='cpu', help='Device for quantization')

    args = parser.parse_args()
    quantize_generator(
        weights_path=args.weights,
        calib_dir=args.calib_dir,
        output_model_path=args.output,
        upscale_factor=args.upscale_factor,
        device_str=args.device
    )
