import os
import argparse
import pickle
import shutil
from tqdm import tqdm


def extract_unseen_images(nih_dir, val_pkl_path, output_dir, max_images=None):
    """
    Extracts images that were not used during training (listed in val_images.pkl)
    from the downloaded NIH_dataset folder and copies them to the output directory.
    """
    if not os.path.exists(nih_dir):
        raise FileNotFoundError(f"NIH dataset directory not found at: {nih_dir}")
    if not os.path.exists(val_pkl_path):
        raise FileNotFoundError(f"Pickle file not found at: {val_pkl_path}")

    # 1. Load the validation/test split filenames
    print(f"[INFO] Loading validation list from: {val_pkl_path}")
    with open(val_pkl_path, 'rb') as f:
        val_paths = pickle.load(f)

    # Convert paths to pure filenames (e.g. '00001255_011.png') for robust matching
    unseen_filenames = {os.path.basename(path) for path in val_paths}
    print(f"[INFO] Found {len(unseen_filenames)} untrained images in the validation split.")

    # 2. Scan the downloaded NIH dataset directory to map available files
    print(f"[INFO] Scanning NIH dataset files in: {nih_dir}...")
    available_files = {}
    for root, _, files in os.walk(nih_dir):
        for file in files:
            if file.endswith('.png'):
                available_files[file] = os.path.join(root, file)

    print(f"[INFO] Found {len(available_files)} total images available in '{nih_dir}'.")

    # 3. Intersect to find which available files are actually in the validation split
    target_files = {name: path for name, path in available_files.items() if name in unseen_filenames}
    print(f"[INFO] Mapped {len(target_files)} available files that were NOT trained.")

    if len(target_files) == 0:
        print("[WARNING] No matching untrained images found. Check if paths or folder structure is correct.")
        return

    # Apply limit if requested
    target_items = list(target_files.items())
    if max_images is not None:
        target_items = target_items[:max_images]
        print(f"[INFO] Limit applied: Extracting only the first {max_images} images.")

    # 4. Copy matched files to output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Copying images to: {output_dir}")

    copied_count = 0
    for name, src_path in tqdm(target_items, desc="Copying unseen images"):
        dst_path = os.path.join(output_dir, name)
        try:
            shutil.copy(src_path, dst_path)
            copied_count += 1
        except Exception as e:
            print(f"[ERROR] Failed to copy {name}: {e}")

    print("\n" + "=" * 60)
    print("         UNSEEN IMAGES EXTRACTION COMPLETED           ")
    print("=" * 60)
    print(f" Source Directory  : {nih_dir}")
    print(f" Target Directory  : {output_dir}")
    print(f" Images Extracted  : {copied_count}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Untrained (Unseen) NIH Chest X-ray Images")
    parser.add_argument('--nih_dir', type=str, default='./NIH_dataset', help='Path to downloaded NIH dataset folder')
    parser.add_argument('--val_pkl', type=str, default='./data/val_images.pkl', help='Path to val_images.pkl file')
    parser.add_argument('--output_dir', type=str, default='./test_images', help='Target folder to save test images')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of extracted images (e.g. 500)')

    args = parser.parse_args()
    extract_unseen_images(
        nih_dir=args.nih_dir,
        val_pkl_path=args.val_pkl,
        output_dir=args.output_dir,
        max_images=args.limit
    )
