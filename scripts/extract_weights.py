import os
import argparse
import torch
import numpy as np

from quantize_model import fuse_generator_bn, replace_conv2d_with_quant, quantize_residual_blocks_only, QuantizedConv2d
from model_architecture import Generator


def extract_quantized_weights(weights_path, output_dir, upscale_factor=4, device_str='cpu'):
    """
    Extracts INT8 weights, scales, zero-points, biases, and activation parameters
    from a Quantized Generator model checkpoint into formatted .txt files for FPGA deployment.
    """
    device = torch.device(device_str)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    print(f"[INFO] Loading quantized model from: {weights_path}")
    checkpoint = torch.load(weights_path, map_location=device)

    is_selective = True
    is_per_channel = True
    state_dict = checkpoint

    if isinstance(checkpoint, dict):
        is_selective = checkpoint.get("selective", True)
        is_per_channel = checkpoint.get("per_channel", True)
        state_dict = checkpoint.get("model", checkpoint)

    # Reconstruct quantized Generator structure
    base_model = Generator(upscale_factor=upscale_factor).to(device)
    fused_model = fuse_generator_bn(base_model)

    if is_selective:
        model = quantize_residual_blocks_only(fused_model, per_channel=is_per_channel)
    else:
        model = fused_model
        replace_conv2d_with_quant(model, per_channel=is_per_channel)

    model.to(device)
    model.load_state_dict(state_dict)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Exporting weights to directory: {output_dir}")

    manifest_lines = []
    manifest_lines.append(f"FPGA ZYNQ INT8 WEIGHT EXPORT MANIFEST")
    manifest_lines.append(f"=" * 60)
    manifest_lines.append(f"Model Checkpoint : {weights_path}")
    manifest_lines.append(f"Selective Mode   : {is_selective}")
    manifest_lines.append(f"Per-Channel Mode : {is_per_channel}")
    manifest_lines.append(f"=" * 60 + "\n")

    layer_count = 0

    for name, module in model.named_modules():
        if isinstance(module, QuantizedConv2d):
            layer_count += 1
            clean_name = name.replace('.', '_')
            prefix = f"layer_{layer_count:02d}_{clean_name}"

            # 1. Extract INT8 Weights
            w_int8 = module.weight_int8.cpu().numpy()
            w_shape = w_int8.shape
            w_txt_path = os.path.join(output_dir, f"{prefix}_weights_int8.txt")
            
            # Save flattened INT8 weights (one integer per line or space-separated)
            np.savetxt(w_txt_path, w_int8.flatten(), fmt='%d')

            # 2. Extract Weight Scales
            w_scale = module.weight_scale.cpu().numpy()
            scale_txt_path = os.path.join(output_dir, f"{prefix}_weight_scale.txt")
            np.savetxt(scale_txt_path, w_scale.flatten(), fmt='%.8e')

            # 3. Extract Biases
            has_bias = module.bias is not None
            if has_bias:
                bias_np = module.bias.cpu().detach().numpy()
                bias_txt_path = os.path.join(output_dir, f"{prefix}_bias_fp32.txt")
                np.savetxt(bias_txt_path, bias_np.flatten(), fmt='%.8e')

            # 4. Extract Activation Scale & Zero Point
            act_in_scale = module.act_in_scale.cpu().item()
            act_in_zp = module.act_in_zero_point.cpu().item()
            act_out_scale = module.act_out_scale.cpu().item()
            act_out_zp = module.act_out_zero_point.cpu().item()

            act_txt_path = os.path.join(output_dir, f"{prefix}_act_params.txt")
            with open(act_txt_path, 'w') as f:
                f.write(f"act_in_scale: {act_in_scale:.8e}\n")
                f.write(f"act_in_zero_point: {act_in_zp}\n")
                f.write(f"act_out_scale: {act_out_scale:.8e}\n")
                f.write(f"act_out_zero_point: {act_out_zp}\n")

            manifest_lines.append(f"Layer {layer_count:02d}: {name}")
            manifest_lines.append(f"  - Module Type  : QuantizedConv2d (INT8)")
            manifest_lines.append(f"  - Weight Shape : {w_shape}")
            manifest_lines.append(f"  - In Channels  : {module.in_channels}")
            manifest_lines.append(f"  - Out Channels : {module.out_channels}")
            manifest_lines.append(f"  - Kernel Size  : {module.kernel_size}")
            manifest_lines.append(f"  - Stride/Pad   : {module.stride} / {module.padding}")
            manifest_lines.append(f"  - Groups       : {module.groups}")
            manifest_lines.append(f"  - Has Bias     : {has_bias}")
            manifest_lines.append(f"  - Files        : {prefix}_weights_int8.txt, {prefix}_weight_scale.txt, {prefix}_act_params.txt")
            manifest_lines.append("-" * 60)

    manifest_path = os.path.join(output_dir, "manifest.txt")
    with open(manifest_path, 'w') as f:
        f.write("\n".join(manifest_lines))

    print("\n" + "=" * 60)
    print(f"       WEIGHT EXTRACTION COMPLETED SUCCESSFULLY       ")
    print("=" * 60)
    print(f" Total Quantized Layers Extracted : {layer_count}")
    print(f" Export Directory                  : {output_dir}")
    print(f" Layer Manifest File               : {manifest_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract INT8 Weights and Parameters for FPGA Zynq")
    parser.add_argument('--weights', type=str, default='./models/netG_4x_quantized_int8_optimized.pth', help='Path to quantized model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./weights_export', help='Output directory for exported .txt files')
    parser.add_argument('--upscale_factor', type=int, default=4, help='Upscale factor')
    parser.add_argument('--device', type=str, default='cpu', help='Device')

    args = parser.parse_args()
    extract_quantized_weights(
        weights_path=args.weights,
        output_dir=args.output_dir,
        upscale_factor=args.upscale_factor,
        device_str=args.device
    )
