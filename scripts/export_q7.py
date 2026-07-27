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

    print(f"[INFO] Loading model from: {weights_path}")
    checkpoint = torch.load(weights_path, map_location=device)
    
    is_qat = False
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        if checkpoint.get("qat") is True:
            is_qat = True
            state_dict = checkpoint["model"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]

    # Check if the state dict matches QAT layout (fused BN, with pointwise bias)
    if any("pointwise.bias" in k for k in state_dict.keys()) and not any("bn.weight" in k for k in state_dict.keys()):
        is_qat = True

    base_model = Generator(upscale_factor=upscale_factor).to(device)

    if is_qat:
        print("[INFO] Detected QAT/Fused checkpoint structure. Restructuring model...")
        fused_model = fuse_generator_bn(base_model)
        # Import the helper to replace Conv2d with Q7QATConv2d
        from train_qat import replace_conv_with_qat
        for res_block in fused_model.residual:
            replace_conv_with_qat(res_block)
        replace_conv_with_qat(fused_model.convblock)
        
        fused_model.load_state_dict(state_dict)
        model_to_export = fused_model
    else:
        print("[INFO] Detected standard FP32 checkpoint structure.")
        base_model.load_state_dict(state_dict)
        base_model.eval()
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
