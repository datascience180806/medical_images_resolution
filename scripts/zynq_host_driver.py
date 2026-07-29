import os
import argparse
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
    Slices an input grayscale image into small non-overlapping tiles (e.g., 64x64).
    
    Args:
        image (np.ndarray): Input grayscale image of shape (H, W).
        tile_size (int): Tile dimensions (default: 64).
        
    Returns:
        tiles (list of np.ndarray): List of image tiles of size (tile_size, tile_size).
        grid_dim (tuple): (num_tiles_h, num_tiles_w) dimensions of grid.
        orig_shape (tuple): (H, W) original image dimensions before padding.
    """
    h, w = image.shape[:2]

    # Calculate padding if dimensions are not divisible by tile_size
    pad_h = (tile_size - (h % tile_size)) % tile_size
    pad_w = (tile_size - (w % tile_size)) % tile_size

    if pad_h > 0 or pad_w > 0:
        image_padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='edge')
    else:
        image_padded = image

    h_pad, w_pad = image_padded.shape
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


def stitch_tiles_into_image(tiles: list, grid_dim: tuple, orig_shape: tuple, scale_factor: int = 2):
    """
    Stitches processed tiles back into a single full-resolution grayscale image.
    
    Args:
        tiles (list of np.ndarray): List of processed tiles of size (128x128).
        grid_dim (tuple): (num_tiles_h, num_tiles_w)
        orig_shape (tuple): (orig_h, orig_w) original image dimensions before padding.
        scale_factor (int): Upscale factor (default: 2 for 64x64 -> 128x128 tiles).
        
    Returns:
        full_image (np.ndarray): Stitched full-resolution grayscale image.
    """
    num_tiles_h, num_tiles_w = grid_dim
    orig_h, orig_w = orig_shape

    tile_sample = tiles[0]
    tile_h, tile_w = tile_sample.shape[:2]

    full_h = num_tiles_h * tile_h
    full_w = num_tiles_w * tile_w

    full_img = np.zeros((full_h, full_w), dtype=tile_sample.dtype)

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


def transfer_dma_pynq(tiles: list, bitstream_path: str = "srcnn_accel.bit", weights_path: str = "srgan_q7_weights_qat.txt", scale_factor: int = 2):
    """
    Transfers tile numpy arrays to Zynq PL Accelerator via AXI DMA using PYNQ framework.
    """
    if not PYNQ_AVAILABLE:
        raise RuntimeError("PYNQ library is not installed. Run on physical Zynq ARM board with PYNQ OS.")

    print(f"[INFO PYNQ] Loading FPGA Bitstream: {bitstream_path}...")
    overlay = Overlay(bitstream_path)
    dma = overlay.axi_dma_0  # Tên kênh DMA cấu hình trên Zynq PL

    # Nạp trọng số xuống FPGA qua DMA nếu đường dẫn được cung cấp
    if weights_path:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Không tìm thấy file trọng số: {weights_path}")
        print(f"[INFO PYNQ] Đang đọc file trọng số Q7 từ {weights_path}...")
        # Đọc file txt, ép kiểu về số nguyên 8-bit có dấu (Int8)
        weights_data = np.loadtxt(weights_path, dtype=np.int8)
        # Cấp phát bộ nhớ đệm PYNQ cho trọng số
        weight_buffer = allocate(shape=weights_data.shape, dtype=np.int8)
        weight_buffer[:] = weights_data
        print(f"[INFO PYNQ] Bắt đầu xả {len(weights_data)} hệ số weights qua AXI-Stream...")
        start_w_time = time.time()
        # Kích hoạt DMA đẩy nguyên một cục weights xuống mạch
        dma.sendchannel.transfer(weight_buffer)
        dma.sendchannel.wait()  # Chờ đẩy xong  
        print(f"[INFO PYNQ] Nạp trọng số thành công! Mất {time.time() - start_w_time:.4f} giây.")
        weight_buffer.freebuffer()

    tile_size = 64
    out_size = tile_size * scale_factor  # 128

    # Cấp phát vùng nhớ liên tục trong RAM để phần cứng đọc/ghi trực tiếp (flat 1D arrays)
    in_buffer = allocate(shape=(tile_size * tile_size,), dtype=np.uint8)
    out_buffer = allocate(shape=(out_size * out_size,), dtype=np.uint8)

    processed_tiles = []
    print(f"[INFO PYNQ] Streaming {len(tiles)} tiles over AXI DMA...")
    start_time = time.time()
    for idx, tile in enumerate(tiles):
        # Đổ mảng Numpy vào buffer đầu vào của DMA
        in_buffer[:] = tile.flatten()

        # Kích hoạt truyền dữ liệu
        dma.sendchannel.transfer(in_buffer)
        dma.recvchannel.transfer(out_buffer)

        # Đợi phần cứng chạy xong
        dma.sendchannel.wait()
        dma.recvchannel.wait()

        # Lấy kết quả ra và reshape lại thành ảnh 2D
        processed_tiles.append(out_buffer.copy().reshape((out_size, out_size)))

    total_time = time.time() - start_time
    print(f"[INFO PYNQ] AXI DMA Transfers completed in {total_time:.4f}s ({total_time/len(tiles)*1000:.2f} ms/tile)")

    # Giải phóng bộ nhớ khi chạy xong
    in_buffer.freebuffer()
    out_buffer.freebuffer()

    return processed_tiles


def simulate_dma_transfer(tiles: list, export_txt_dir: str = None, scale_factor: int = 2):
    """
    Simulates AXI DMA transfers for PC / Kaggle testing or exports tile arrays to TXT files.
    """
    print(f"[SIMULATION] Simulating AXI DMA Transfer for {len(tiles)} tiles (Tile size: 64x64 -> 128x128)...")
    
    if export_txt_dir:
        os.makedirs(export_txt_dir, exist_ok=True)
        print(f"[SIMULATION] Exporting tile array TXT files to: {export_txt_dir}")

    processed_tiles = []
    for idx, tile in enumerate(tiles):
        # 1. Export tile array to TXT format for Verilog/VHDL testbench (flat values, one per line)
        if export_txt_dir:
            txt_path = os.path.join(export_txt_dir, f"tile_{idx:04d}_in_64x64.txt")
            np.savetxt(txt_path, tile.flatten(), fmt='%d')

        # 2. Simulate upscaling by 2 using standard Bicubic for output simulation
        tile_h, tile_w = tile.shape[:2]
        out_h, out_w = tile_h * scale_factor, tile_w * scale_factor
        upscaled_tile = cv2.resize(tile, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        processed_tiles.append(upscaled_tile)

    return processed_tiles


def main():
    parser = argparse.ArgumentParser(description="Zynq PS Python Host: OpenCV Grayscale Tiling & AXI DMA Driver")
    parser.add_argument('--image_path', type=str, default='./assets/sample_lr_input.png', help='Path to input grayscale X-ray image')
    parser.add_argument('--tile_size', type=int, default=64, help='Tile crop dimensions (fixed at 64x64)')
    parser.add_argument('--scale_factor', type=int, default=2, help='Upscale factor (fixed at 2)')
    parser.add_argument('--output_path', type=str, default='./assets/output_zynq_sr.png', help='Output stitched image path')
    parser.add_argument('--bitstream', type=str, default='srcnn_accel.bit', help='Path to Vivado FPGA bitstream file')
    parser.add_argument('--weights_path', type=str, default='srgan_q7_weights_qat.txt', help='Path to Q7 weights file')
    parser.add_argument('--export_txt_dir', type=str, default='./dma_txt_buffers', help='Directory to export TXT tile buffers')
    parser.add_argument('--sim', action='store_true', help='Force Simulation Mode (no PYNQ required)')

    args = parser.parse_args()

    # 1. Read input X-ray image in Grayscale as required
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Input image not found: {args.image_path}")

    print(f"[HOST] Reading input image in Grayscale: {args.image_path}")
    img_gray = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)

    orig_h, orig_w = img_gray.shape[:2]
    print(f"[HOST] Input image resolution: {orig_w}x{orig_h} (Grayscale)")

    # 2. Slice image into 64x64 tiles
    print(f"[HOST] Slicing image into {args.tile_size}x{args.tile_size} tiles...")
    tiles, grid_dim, orig_dim = slice_image_into_tiles(img_gray, tile_size=args.tile_size)
    print(f"[HOST] Generated {len(tiles)} tiles ({grid_dim[0]}x{grid_dim[1]} grid)")

    # 3. Transfer tiles over AXI DMA (PYNQ Hardware or Simulation Mode)
    if PYNQ_AVAILABLE and not args.sim:
        print("[HOST] Running on Zynq PS ARM board (PYNQ Mode)...")
        processed_tiles = transfer_dma_pynq(
            tiles, 
            bitstream_path=args.bitstream, 
            weights_path=args.weights_path, 
            scale_factor=args.scale_factor
        )
    else:
        print("[HOST] Running in Simulation/Kaggle Mode...")
        processed_tiles = simulate_dma_transfer(tiles, export_txt_dir=args.export_txt_dir, scale_factor=args.scale_factor)

    # 4. Stitch tiles back into Super-Resolution image (128x128 per tile)
    print(f"[HOST] Stitching processed tiles back into full image (Scale factor: x{args.scale_factor})...")
    sr_image = stitch_tiles_into_image(processed_tiles, grid_dim, orig_dim, scale_factor=args.scale_factor)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    cv2.imwrite(args.output_path, sr_image)

    print("\n" + "=" * 60)
    print(f"       ZYNQ HOST DRIVER EXECUTED SUCCESSFULLY       ")
    print("=" * 60)
    print(f" Output Resolution : {sr_image.shape[1]}x{sr_image.shape[0]} (Grayscale)")
    print(f" Output Image Saved : {args.output_path}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
