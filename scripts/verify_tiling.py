import os
import cv2
import numpy as np
from zynq_host_driver import slice_image_into_tiles, stitch_tiles_into_image


def test_tiling_exactness(image_path, tile_size=64):
    """
    Tests whether the tiling and stitching logic in zynq_host_driver.py is 100% pixel-exact.
    """
    if not os.path.exists(image_path):
        print(f"[ERROR] Test image not found at: {image_path}")
        return

    print("=" * 60)
    print("       ROUND-TRIP TILING & STITCHING VERIFICATION TEST      ")
    print("=" * 60)
    print(f"[TEST] Loading test image: {image_path}")
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    orig_h, orig_w, channels = img_rgb.shape
    print(f"[TEST] Original Image Dimensions: {orig_w}x{orig_h} ({channels} channels)")

    # 1. Slice into tiles (scale_factor = 1 for identity round-trip test)
    print(f"[TEST] Slicing image into {tile_size}x{tile_size} tiles...")
    tiles, grid_dim, orig_dim = slice_image_into_tiles(img_rgb, tile_size=tile_size)
    print(f"[TEST] Total tiles generated: {len(tiles)} ({grid_dim[0]}x{grid_dim[1]} grid)")

    # 2. Stitch tiles back without modifying them (scale_factor = 1)
    print("[TEST] Re-stitching tiles back into full image...")
    reconstructed_rgb = stitch_tiles_into_image(tiles, grid_dim, orig_dim, scale_factor=1)

    # 3. Calculate absolute pixel difference
    diff = np.abs(img_rgb.astype(np.int32) - reconstructed_rgb.astype(np.int32))
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    mse = np.mean((img_rgb.astype(np.float64) - reconstructed_rgb.astype(np.float64)) ** 2)

    print("\n" + "-" * 60)
    print(f" Reconstruction Shape : {reconstructed_rgb.shape[1]}x{reconstructed_rgb.shape[0]}")
    print(f" Max Pixel Difference : {max_diff} (Target: 0)")
    print(f" Mean Pixel Difference: {mean_diff:.6f} (Target: 0.0)")
    print(f" Mean Squared Error   : {mse:.6f} (Target: 0.0)")
    print("-" * 60)

    if max_diff == 0 and mse == 0.0:
        print("\nSUCCESS: Tiling and Stitching logic is 100% PIXEL-EXACT!")
        print("File Host hoat dong chinh xac 100% tuyet doi, khong mat mát du chi 1 pixel!")
    else:
        print("\nFAILURE: Discrepancy detected between original and reconstructed image.")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    test_image = "./assets/sample_lr_input.png"
    test_tiling_exactness(test_image, tile_size=64)
