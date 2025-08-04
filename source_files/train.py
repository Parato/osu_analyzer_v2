# train.py

# This script handles the training of the YOLOv8 model.
# MODIFIED FOR VERTEX AI:
# - Added logic to upload the final training results (the 'runs' directory)
#   from the local container environment to a persistent GCS location.
# - YOLOv8 natively supports reading data from GCS, so no changes are
#   needed for the input paths (`--data`).

from ultralytics import YOLO
import torch
import argparse
import os
import subprocess
from urllib.parse import urlparse


def upload_directory_to_gcs(local_path, gcs_path):
    """
    Uploads a directory from the local filesystem to GCS using gsutil.
    This is efficient for uploading the entire 'runs' directory.
    """
    if not gcs_path.startswith('gs://'):
        print(f"[ERROR] GCS path must start with gs://. Got: {gcs_path}")
        return

    print(f"Uploading results from '{local_path}' to '{gcs_path}'...")
    # Using gsutil for recursive copy is robust and fast.
    # The Docker container will have gcloud SDK installed.
    command = ["gsutil", "-m", "rsync", "-r", local_path, gcs_path]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Successfully uploaded results to {gcs_path}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to upload results to GCS.")
        print(f"Command: {' '.join(command)}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
    except FileNotFoundError:
        print("[ERROR] 'gsutil' command not found. Make sure the Google Cloud SDK is installed in the container.")


def main():
    """
    Main function to start the model training process.
    """
    # In the Docker container, the code is placed in /app
    SCRIPT_DIR = "/app"

    parser = argparse.ArgumentParser(description="YOLOv8 Model Trainer for Vertex AI")
    # --data can be a local path or a GCS path (gs://...). YOLO handles it.
    parser.add_argument("--data", required=True, help="Path to the dataset YAML configuration file (local or GCS).")
    # --weights can also be a local or GCS path.
    parser.add_argument("--weights", default="yolov8s.pt",
                        help="Path to initial weights (.pt file). Can be 'yolov8s.pt' or a GCS path.")
    # --project will be a LOCAL path inside the container. We upload it later.
    parser.add_argument("--project", default=os.path.join(SCRIPT_DIR, "runs", "detect"),
                        help="Local directory to save training runs inside the container.")
    parser.add_argument("--name", default="yolov8s_osu_custom", help="Name for the training run directory.")
    parser.add_argument("--batch-size", type=int, default=16, help="Set the training batch size.")
    # New argument to specify the GCS output location for the final results
    parser.add_argument("--gcs-output-path", required=True, help="GCS path to upload the final 'runs' directory to.")

    args = parser.parse_args()

    print("--- Osu! Detector Model Training on Vertex AI ---")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cpu':
        print("[WARN] No GPU detected. Training will be very slow.")

    # Check if weights path is GCS, if so, YOLO will handle the download
    if args.weights.startswith('gs://'):
        print(f"Loading initial weights from GCS: {args.weights}")
    else:
        # If it's a local path, make sure it's absolute or relative to the script dir
        if not os.path.isabs(args.weights):
            args.weights = os.path.join(SCRIPT_DIR, args.weights)
        print(f"Loading initial weights from local path: {args.weights}")
        if not os.path.exists(args.weights):
            print(f"[INFO] Weights file not found at '{args.weights}'. YOLO will attempt to download it.")

    model = YOLO(args.weights)
    model.to(device)

    print(f"Starting model training using dataset config: {args.data}")
    print("This may take a long time depending on your dataset size and hardware.")
    print(f"Local results will be saved in: {os.path.join(args.project, args.name)}")

    try:
        results = model.train(
            data=args.data,
            epochs=50,
            imgsz=640,
            patience=10,
            project=args.project,
            name=args.name,
            batch=args.batch_size,

            # --- Online Augmentation Parameters ---
            degrees=5.0,
            translate=0.1,
            scale=0.5,
            shear=2.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            mosaic=1.0,
            mixup=0.1,
        )
    except Exception as e:
        print(f"\n[ERROR] An error occurred during training: {e}")
        print("[ERROR] Please ensure your dataset paths in the YAML file are correct")
        print("[ERROR] and that you have permissions to access them.")
        # After a failure, still try to upload any partial results for debugging
        print("Attempting to upload partial results for debugging...")
        local_results_dir = os.path.join(args.project, args.name)
        if os.path.exists(local_results_dir):
            upload_directory_to_gcs(args.project, args.gcs_output_path)
        return

    print("\n--- Training Complete ---")

    # --- MODIFIED: Upload results to GCS ---
    # The results are saved in args.project, which defaults to './runs/detect'
    # We upload the entire project directory.
    print("Uploading training results to Google Cloud Storage...")
    upload_directory_to_gcs(args.project, args.gcs_output_path)

    print("\nNext Steps:")
    print(f"1. Check the GCS bucket '{args.gcs_output_path}' for your training results.")
    print("2. Locate the new 'best.pt' file in the output directory.")
    print("3. Use this new model for inference.")


if __name__ == '__main__':
    main()