import json
import os
import pandas as pd

def main():
    json_path = "./benchmark_2200_checkpoint.json"
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    results = data.get("per_image_results", [])
    print(f"Total image records read: {len(results)}")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Group by dataset and compute mean
    grouped = df.groupby("dataset").agg(
        image_count=("filename", "count"),
        avg_latency_ms=("latency_ms", "mean"),
        avg_psnr_bicubic=("psnr_bicubic_db", "mean"),
        avg_ssim_bicubic=("ssim_bicubic", "mean"),
        avg_psnr_fpga=("psnr_fpga_db", "mean"),
        avg_ssim_fpga=("ssim_fpga", "mean"),
        avg_psnr_gain=("psnr_gain_db", "mean")
    ).reset_index()

    # Print results manually
    print("\n=========================================================================")
    print("                 AVERAGE METRICS FOR EACH DATASET                         ")
    print("=========================================================================")
    print(f"{'Dataset':<15} | {'Count':<5} | {'Latency (ms)':<12} | {'PSNR Bic (dB)':<13} | {'SSIM Bic':<8} | {'PSNR FPGA':<9} | {'SSIM FPGA':<9} | {'Gain (dB)':<9}")
    print("-" * 105)
    for _, row in grouped.iterrows():
        print(f"{row['dataset']:<15} | {int(row['image_count']):<5} | {row['avg_latency_ms']:<12.3f} | {row['avg_psnr_bicubic']:<13.3f} | {row['avg_ssim_bicubic']:<8.4f} | {row['avg_psnr_fpga']:<9.3f} | {row['avg_ssim_fpga']:<9.4f} | {row['avg_psnr_gain']:<9.3f}")
    print("=========================================================================\n")


if __name__ == "__main__":
    main()
