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
    
    if conv.groups > 1 and conv.groups == conv.in_channels:
        # Depthwise conv weight shape: (in_channels, 1, K, K)
        t_conv = (gamma / std).reshape(-1, 1, 1, 1)
    else:
        # Standard or pointwise conv weight shape: (out_channels, in_channels/groups, K, K)
        t_conv = (gamma / std).reshape(-1, 1, 1, 1)

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
            block.cnn.pointwise = fuse_conv_bn_eval(block.cnn.pointwise, block.bn)
            block.bn = nn.Identity()

    _fuse_conv_block(net.initial)

    for res_block in net.residual:
        _fuse_conv_block(res_block.block1)
        _fuse_conv_block(res_block.block2)

    _fuse_conv_block(net.convblock)

    return net


class QuantizedConv2d(nn.Module):
    """
    Wrapper layer for Conv2d supporting Per-Channel INT8 weight quantization
    and tracking activation statistics.
    """
    def __init__(self, conv: nn.Conv2d, num_bits=8, per_channel=True):
        super(QuantizedConv2d, self).__init__()
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.groups = conv.groups
        self.num_bits = num_bits
        self.per_channel = per_channel

        w_fp32 = conv.weight.data.clone()

        if per_channel:
            # Per-Channel Quantization: Scale for each output channel [out_channels, 1, 1, 1]
            w_flat = w_fp32.view(self.out_channels, -1)
            max_abs = torch.max(torch.abs(w_flat), dim=1)[0]
            scale_w = torch.where(max_abs == 0, torch.tensor(1.0, device=w_fp32.device), max_abs / 127.0)
            scale_w_view = scale_w.view(-1, 1, 1, 1)

            w_int8 = torch.clamp(torch.round(w_fp32 / scale_w_view), -127, 127).to(torch.int8)
            self.w_dequantized = (w_int8.float() * scale_w_view)
        else:
            # Per-Tensor Quantization
            w_min = w_fp32.min().item()
            w_max = w_fp32.max().item()
            max_abs = max(abs(w_min), abs(w_max))
            scale_w = 1.0 if max_abs == 0 else max_abs / 127.0
            scale_w_view = torch.tensor(scale_w, dtype=torch.float32)

            w_int8 = torch.clamp(torch.round(w_fp32 / scale_w), -127, 127).to(torch.int8)
            self.w_dequantized = (w_int8.float() * scale_w)

        self.register_buffer("weight_int8", w_int8)
        self.register_buffer("weight_scale", scale_w_view)
        self.register_buffer("weight_zero_point", torch.zeros_like(scale_w_view, dtype=torch.int32))

        if conv.bias is not None:
            self.register_buffer("bias", conv.bias.data.clone())
        else:
            self.bias = None

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

        out = nn.functional.conv2d(
            x, self.w_dequantized, self.bias,
            stride=self.stride, padding=self.padding, groups=self.groups
        )

        if self.calibrating:
            self.update_act_stats(out, self.act_out_min, self.act_out_max)

        return out


def replace_conv2d_with_quant(module, per_channel=True):
    """
    Recursively replaces all nn.Conv2d layers with QuantizedConv2d.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            setattr(module, name, QuantizedConv2d(child, per_channel=per_channel))
        else:
            replace_conv2d_with_quant(child, per_channel=per_channel)


def quantize_residual_blocks_only(generator, per_channel=True):
    """
    Selectively quantizes ONLY the 16 Residual Blocks and Intermediate ConvBlock (FPGA Acceleration target),
    preserving Head (initial) and Tail (upsampler, final_conv) in FP32 precision.
    """
    net = copy.deepcopy(generator)
    
    # Quantize only 16 Residual Blocks
    print("[INFO] Quantizing 16 Residual Blocks...")
    for res_block in net.residual:
        replace_conv2d_with_quant(res_block, per_channel=per_channel)

    # Quantize Intermediate ConvBlock
    print("[INFO] Quantizing Intermediate ConvBlock...")
    replace_conv2d_with_quant(net.convblock, per_channel=per_channel)

    return net


def quantize_generator(weights_path, calib_dir, output_model_path, upscale_factor=4, selective=True, per_channel=True, device_str='cpu'):
    """
    Main PTQ Quantization workflow.
    
    Args:
        selective (bool): If True, preserves Head & Tail in FP32 and quantizes 16 Residual Blocks (FPGA PL target).
        per_channel (bool): If True, uses Per-Channel weight quantization for Depthwise Conv layers.
    """
    device = torch.device(device_str)
    print(f"[INFO] Running Quantization process on device: {device}")
    print(f"[CONFIG] Selective (Residual-only) Quantization: {selective}")
    print(f"[CONFIG] Per-Channel Weight Quantization       : {per_channel}")

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

    # 3. Quantize layers
    if selective:
        print("[INFO] Performing Selective Quantization (Preserving Head & Tail FP32, Quantizing Residual Blocks)...")
        quant_net = quantize_residual_blocks_only(fused_net, per_channel=per_channel)
    else:
        print("[INFO] Performing Full-Model Quantization...")
        quant_net = copy.deepcopy(fused_net)
        replace_conv2d_with_quant(quant_net, per_channel=per_channel)

    quant_net.to(device)

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

            _ = quant_net(lr_tensor)

    # 5. Finalize activation scales and zero points
    print("[INFO] Finalizing INT8 Quantization scales & zero points...")
    for m in quant_net.modules():
        if isinstance(m, QuantizedConv2d):
            m.finalize_quantization()

    # 6. Save Quantized Model
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    torch.save({"model": quant_net.state_dict(), "quantized": True, "selective": selective, "per_channel": per_channel}, output_model_path)
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
    parser.add_argument('--selective', action='store_true', default=True, help='Preserve Head & Tail in FP32, quantize 16 Residual Blocks')
    parser.add_argument('--full', action='store_true', help='Quantize all layers including Head & Tail')
    parser.add_argument('--per_channel', action='store_true', default=True, help='Use Per-Channel weight quantization for Depthwise Conv')
    parser.add_argument('--device', type=str, default='cpu', help='Device for quantization')

    args = parser.parse_args()
    
    selective_mode = not args.full

    quantize_generator(
        weights_path=args.weights,
        calib_dir=args.calib_dir,
        output_model_path=args.output,
        upscale_factor=args.upscale_factor,
        selective=selective_mode,
        per_channel=args.per_channel,
        device_str=args.device
    )
