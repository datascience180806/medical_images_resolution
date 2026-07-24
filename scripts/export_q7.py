import os
import argparse
import torch
import numpy as np

from model_architecture import Generator
from quantize_model import fuse_generator_bn


def quantize_to_q7(tensor):
    """
    Quantizes a PyTorch tensor to Q7 format:
    Formula: Q7 = Clamp( Round( Float32 * 128 ), -128, 127 )
    """
    scaled_tensor = tensor * 128.0
    rounded_tensor = torch.round(scaled_tensor)
    q7_tensor = torch.clamp(rounded_tensor, min=-128, max=127)
    return q7_tensor.detach().cpu().numpy().astype(np.int8)


def export_q7_weights(weights_path, output_filename="srgan_q7_weights.txt", upscale_factor=4, fuse_bn=True):
    """
    Loads FP32 model, optionally fuses BN layers, quantizes all weights/biases to Q7 format,
    and exports them to a single flat .txt file as requested by the hardware constraints.
    """
    device = torch.device('cpu')
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    print(f"[INFO] Loading FP32 model from: {weights_path}")
    base_model = Generator(upscale_factor=upscale_factor).to(device)
    checkpoint = torch.load(weights_path, map_location=device)
    
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        base_model.load_state_dict(checkpoint["model"])
    else:
        base_model.load_state_dict(checkpoint)
    base_model.eval()

    # 1. Perform BatchNorm Fusion if requested (vital for hardware compatibility)
    if fuse_bn:
        print("[INFO] Fusing Conv2d + BatchNorm2d layers before quantization...")
        model_to_export = fuse_generator_bn(base_model)
    else:
        model_to_export = base_model

    print(f"[INFO] Exporting all parameters to Q7 format in: {output_filename}...")
    
    total_elements = 0
    exported_layers = []

    with open(output_filename, 'w') as f:
        # Loop through all parameters (weights & biases) of the Generator
        for name, param in model_to_export.named_parameters():
            # Skip parameters belonging to Identity/fused BN structures if they were kept
            if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name:
                continue
            
            if 'weight' in name or 'bias' in name:
                # Apply the requested Q7 quantization formula
                q7_array = quantize_to_q7(param).flatten()
                
                # Write to the single unified txt file
                for val in q7_array:
                    f.write(f"{val}\n")
                
                total_elements += len(q7_array)
                exported_layers.append(f"{name} ({param.shape} -> {len(q7_array)} elements)")

    print("\n" + "=" * 60)
    print("        Q7 WEIGHT EXPORT COMPLETED SUCCESSFULLY       ")
    print("=" * 60)
    print(f" Output File       : {output_filename}")
    print(f" Total Q7 Values   : {total_elements}")
    print(f" Exported Layers   :\n  - " + "\n  - ".join(exported_layers))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Model Weights in Q7 format for FPGA")
    parser.add_argument('--weights', type=str, default='./models/netG_4x_epoch5.pth.tar', help='Path to FP32 weights')
    parser.add_argument('--output', type=str, default='./models/srgan_q7_weights.txt', help='Output Q7 file path')
    parser.add_argument('--upscale_factor', type=int, default=4, help='Upscale factor')
    parser.add_argument('--no_fuse', action='store_true', help='Do not fuse BN layers before quantization')

    args = parser.parse_args()
    export_q7_weights(
        weights_path=args.weights,
        output_filename=args.output,
        upscale_factor=args.upscale_factor,
        fuse_bn=not args.no_fuse
    )
