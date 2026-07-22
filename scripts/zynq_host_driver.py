import os
import argparse
import math
import time
import numpy as np
import cv2

# Check if PYNQ library is available (on physical Zynq PS ARM board)
try:
    from pynq import Overlay, allocate
    PYNQ_AVAILABLE = True
except ImportError:
    PYNQ_AVAILABLE = False


def slice_image_into_tiles(image: np.ndarray, tile_size: int = 64):
    """
    Slices an input OpenCV image into small non-overlapping tiles (e.g., 64x64).
    
    Args:
        image (np.ndarray): Input OpenCV image of shape (H, W, C) or (H, W).
        tile_size (int): Tile dimensions (default: 64).
        
    Returns:
        tiles (list of np.ndarray): List of image tiles of size (tile_size, tile_size, C).
        grid_dim (tuple): (num_tiles_h, num_tiles_w) dimensions of grid.
        padded_shape (tuple): (H_padded, W_padded) after padding to tile_size multiples.
    """
    if len(image.shape) == 2:
        h, w = image.shape
        c = 1
    else:
        h, w, c = image.shape

    # Calculate padding if dimensions are not divisible by tile_size
    pad_h = (tile_size - (h % tile_size)) % tile_size
    pad_w = (tile_size - (w % tile_size)) % tile_size

    if pad_h > 0 or pad_w > 0:
        if c == 1:
            image_padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='edge')
        else:
            image_padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
    else:
        image_padded = image

    h_pad, w_pad = image_padded.shape[:2]
    num_tiles_h = h_pad // tile_size
    num_tiles_w = w_pad // tile_size

    tiles = []
    for i in range(num_tiles_h):
        for j in range(num_tiles_w):
            y_start = i * tile_size
            y_end = y_start + tile_size
            x_start = j * tile_size
            x_end = x_start + tile_size

            tile = image_padded[y_start:y_end, x_start:x_end]
            tiles.append(tile)

    return tiles, (num_tiles_h, num_tiles_w), (h, w)


def stitch_tiles_into_image(tiles: list, grid_dim: tuple, orig_shape: tuple, scale_factor: int = 4):
    """
    Stitches processed tiles back into a single full-resolution image.
    
    Args:
        tiles (list of np.ndarray): List of processed tiles.
        grid_dim (tuple): (num_tiles_h, num_tiles_w)
        orig_shape (tuple): (orig_h, orig_w) original image dimensions before padding.
        scale_factor (int): Upscale factor (default: 4 for 64x64 -> 256x256 tiles).
        
    Returns:
        full_image (np.ndarray): Stitched full-resolution image.
    """
    num_tiles_h, num_tiles_w = grid_dim
    orig_h, orig_w = orig_shape

    tile_sample = tiles[0]
    tile_h, tile_w = tile_sample.shape[:2]

    full_h = num_tiles_h * tile_h
    full_w = num_tiles_w * tile_w

    if len(tile_sample.shape) == 2:
        full_img = np.zeros((full_h, full_w), dtype=tile_sample.dtype)
    else:
        c = tile_sample.shape[2]
        full_img = np.zeros((full_h, full_w, c), dtype=tile_sample.dtype)

    idx = 0
    for i in range(num_tiles_h):
        for j in range(num_tiles_w):
            y_start = i * tile_h
            y_end = y_start + tile_h
            x_start = j * tile_w
            x_end = x_start + tile_w

            full_img[y_start:y_end, x_start:x_end] = tiles[idx]
            idx += 1

    # Crop out any padding added during slicing (scaled by scale_factor)
    final_h = orig_h * scale_factor
    final_w = orig_w * scale_factor
    full_img_cropped = full_img[:final_h, :final_w]

    return full_img_cropped


def transfer_dma_pynq(tiles: list, bitstream_path: str = None, scale_factor: int = 4):
    """
    Transfers tile numpy arrays to Zynq PL Accelerator via AXI DMA using PYNQ framework.
    """
    if not PYNQ_AVAILABLE:
        raise RuntimeError("PYNQ library is not installed. Run on physical Zynq ARM board with PYNQ OS.")

    print(f"[INFO PYNQ] Loading FPGA Bitstream: {bitstream_path}...")
    overlay = Overlay(bitstream_path)
    dma = overlay.axi_dma_0  # Assumes AXI DMA IP name in Vivado IP Integrator

    tile_sample = tiles[0]
    tile_h, tile_w = tile_sample.shape[:2]
    out_h, out_w = tile_h * scale_factor, tile_w * scale_factor
    channels = tile_sample.shape[2] if len(tile_sample.shape) == 3 else 1

    # Allocate contiguous memory buffers for AXI DMA
    in_buffer = allocate(shape=(tile_h, tile_w, channels), dtype=np.uint8)
    out_buffer = allocate(shape=(out_h, out_w, channels), dtype=np.uint8)

    processed_tiles = []
    print(f"[INFO PYNQ] Streaming {len(tiles)} tiles over AXI DMA...")
    start_time = time.time()

    for idx, tile in enumerate(tiles):
        in_buffer[:] = tile

        # Initiate AXI DMA Transfer
        dma.sendchannel.transfer(in_buffer)
        dma.recvchannel.transfer(out_buffer)

        # Wait for hardware accelerator in PL to finish
        dma.sendchannel.wait()
        dma.recvchannel.wait()

        processed_tiles.append(out_buffer.copy())

    total_time = time.time() - start_time
    print(f"[INFO PYNQ] AXI DMA Transfers completed in {total_time:.4f}s ({total_time/len(tiles)*1000:.2f} ms/tile)")

    return processed_tiles


def simulate_dma_transfer(tiles: list, export_txt_dir: str = None, scale_factor: int = 4):
    """
    Simulates AXI DMA transfers for PC / Kaggle testing or exports tile arrays to TXT files.
    """
    print(f"[SIMULATION] Simulating AXI DMA Transfer for {len(tiles)} tiles (Tile size: {tiles[0].shape[:2]})...")
    
    if export_txt_dir:
        os.makedirs(export_txt_dir, exist_ok=True)
        print(f"[SIMULATION] Exporting tile array TXT files to: {export_txt_dir}")

    processed_tiles = []
    for idx, tile in enumerate(tiles):
        # 1. Optionally export tile array to TXT format for Verilog/VHDL testbench
        if export_txt_dir:
            txt_path = os.path.join(export_txt_dir, f"tile_{idx:04d}_in_64x64.txt")
            np.savetxt(txt_path, tile.flatten(), fmt='%d')

        # 2. Simulate Bicubic 4x upscaling in simulation mode
        tile_h, tile_w = tile.shape[:2]
        out_h, out_w = tile_h * scale_factor, tile_w * scale_factor
        upscaled_tile = cv2.resize(tile, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        processed_tiles.append(upscaled_tile)

    return processed_tiles


def main():
    parser = argparse.ArgumentParser(description="Zynq PS Python Host: OpenCV Tiling & AXI DMA Driver")
    parser.add_argument('--image_path', type=str, default='./assets/sample_lr_input.png', help='Path to input X-ray image')
    parser.add_argument('--tile_size', type=int, default=64, help='Tile crop dimensions (default: 64x64)')
    parser.add_argument('--scale_factor', type=int, default=4, help='Upscale factor (default: 4)')
    parser.add_argument('--output_path', type=str, default='./assets/output_zynq_sr.png', help='Output stitched image path')
    parser.add_argument('--bitstream', type=str, default='./srgan_accelerator.bit', help='Path to Vivado FPGA bitstream file')
    parser.add_argument('--export_txt_dir', type=str, default='./dma_txt_buffers', help='Directory to export TXT tile buffers')
    parser.add_argument('--sim', action='store_true', help='Force Simulation Mode (no PYNQ required)')

    args = parser.parse_args()

    # 1. Read input X-ray image using OpenCV
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Input image not found: {args.image_path}")

    print(f"[HOST] Reading input image using OpenCV: {args.image_path}")
    img_bgr = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    orig_h, orig_w = img_rgb.shape[:2]
    print(f"[HOST] Input image resolution: {orig_w}x{orig_h} ({img_rgb.shape[2]} channels)")

    # 2. Slice image into 64x64 tiles
    print(f"[HOST] Slicing image into {args.tile_size}x{args.tile_size} tiles...")
    tiles, grid_dim, orig_dim = slice_image_into_tiles(img_rgb, tile_size=args.tile_size)
    print(f"[HOST] Generated {len(tiles)} tiles ({grid_dim[0]}x{grid_dim[1]} grid)")

    # 3. Transfer tiles over AXI DMA (PYNQ Hardware or Simulation Mode)
    if PYNQ_AVAILABLE and not args.sim:
        print("[HOST] Running on Zynq PS ARM board (PYNQ Mode)...")
        processed_tiles = transfer_dma_pynq(tiles, bitstream_path=args.bitstream, scale_factor=args.scale_factor)
    else:
        print("[HOST] Running in Simulation/Kaggle Mode...")
        processed_tiles = simulate_dma_transfer(tiles, export_txt_dir=args.export_txt_dir, scale_factor=args.scale_factor)

    # 4. Stitch tiles back into Super-Resolution image
    print(f"[HOST] Stitching processed tiles back into full image (Scale factor: x{args.scale_factor})...")
    sr_image_rgb = stitch_tiles_into_image(processed_tiles, grid_dim, orig_dim, scale_factor=args.scale_factor)

    sr_image_bgr = cv2.cvtColor(sr_image_rgb, cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    cv2.imwrite(args.output_path, sr_image_bgr)

    print("\n" + "=" * 60)
    print(f"       ZYNQ HOST DRIVER EXECUTED SUCCESSFULLY       ")
    print("=" * 60)
    print(f" Output Resolution : {sr_image_rgb.shape[1]}x{sr_image_rgb.shape[0]}")
    print(f" Output Image Saved : {args.output_path}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
