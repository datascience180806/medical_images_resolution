import argparse
import hashlib
import os
import torch
import numpy as np

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description="Export PyTorch SRCNN weights to Q7/Q14 hex files.")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to .pth checkpoint file")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")
    parser.add_argument("--dummy", action="store_true", help="Generate synthetic non-zero weights if no checkpoint")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    weights_path = os.path.join(args.output_dir, "weights_hex_clean.txt")
    biases_path = os.path.join(args.output_dir, "biases_hex_clean.txt")

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

        # Strip module. prefix if trained with DataParallel
        clean_state = {}
        for k, v in state_dict.items():
            key = k.replace("module.", "")
            clean_state[key] = v

        # Extract weights and biases
        w1 = clean_state["conv1.weight"].numpy() # [16, 1, 9, 9]
        b1 = clean_state["conv1.bias"].numpy()   # [16]
        w2 = clean_state["conv2.weight"].numpy() # [8, 16, 1, 1]
        b2 = clean_state["conv2.bias"].numpy()   # [8]
        w3 = clean_state["conv3.weight"].numpy() # [1, 8, 5, 5]
        b3 = clean_state["conv3.bias"].numpy()   # [1]
    else:
        print("No checkpoint provided or file not found. Generating default synthetic weights...")
        # Synthetic non-zero identity/pass-through weights
        w1 = np.zeros((16, 1, 9, 9), dtype=np.float32)
        w1[:, 0, 4, 4] = 0.5
        b1 = np.zeros(16, dtype=np.float32)

        w2 = np.zeros((8, 16, 1, 1), dtype=np.float32)
        for i in range(8):
            w2[i, i, 0, 0] = 0.5
            w2[i, i+8, 0, 0] = 0.5
        b2 = np.zeros(8, dtype=np.float32)

        w3 = np.zeros((1, 8, 5, 5), dtype=np.float32)
        w3[0, :, 2, 2] = 0.25
        b3 = np.zeros(1, dtype=np.float32)

    # Flatten tensors
    all_weights = np.concatenate([w1.flatten(), w2.flatten(), w3.flatten()])
    all_biases = np.concatenate([b1.flatten(), b2.flatten(), b3.flatten()])

    assert len(all_weights) == 1624, f"Weight count mismatch: expected 1624, got {len(all_weights)}"
    assert len(all_biases) == 25, f"Bias count mismatch: expected 25, got {len(all_biases)}"

    # Quantization: Q7 for weights (x128), Q14 for biases (x16384)
    q7_weights = np.clip(np.round(all_weights * 128.0), -128, 127).astype(np.int8)
    q14_biases = np.clip(np.round(all_biases * 16384.0), -2147483648, 2147483647).astype(np.int32)

    # Export weights_hex_clean.txt (8-bit hex per line)
    with open(weights_path, "w") as f:
        for val in q7_weights:
            hex_val = f"{int(val) & 0xFF:02X}"
            f.write(f"{hex_val}\n")

    # Export biases_hex_clean.txt (32-bit hex per line)
    with open(biases_path, "w") as f:
        for val in q14_biases:
            hex_val = f"{int(val) & 0xFFFFFFFF:08X}"
            f.write(f"{hex_val}\n")

    print(f"Exported: {weights_path} (1624 signed Q7 weights, SHA-256: {compute_sha256(weights_path)})")
    print(f"Exported: {biases_path} (25 signed Q14 biases, SHA-256: {compute_sha256(biases_path)})")

if __name__ == "__main__":
    main()
