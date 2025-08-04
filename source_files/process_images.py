# Usage
# python process_images.py pictures backgrounds


import os
import argparse
from PIL import Image
from tqdm import tqdm


def process_images(input_dir, output_dir, target_size=(1280, 720), quality=80):
    """
    Resizes and converts images from an input directory and saves them to an
    output directory.

    Args:
        input_dir (str): The path to the directory containing the original images.
        output_dir (str): The path to the directory where processed images will be saved.
        target_size (tuple): The target (width, height) for the output images.
        quality (int): The JPEG quality for the output images (1-100).
    """
    # --- 1. Validate paths and create output directory if it doesn't exist ---
    if not os.path.isdir(input_dir):
        print(f"[ERROR] Input directory not found at: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Processed images will be saved to: {output_dir}")

    # --- 2. Find all valid image files ---
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    image_files = [
        f for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(valid_extensions)
    ]

    if not image_files:
        print(f"[WARN] No images found in {input_dir}")
        return

    print(f"[INFO] Found {len(image_files)} images to process.")

    # --- 3. Process each image with a progress bar ---
    for filename in tqdm(image_files, desc="Processing Images"):
        input_path = os.path.join(input_dir, filename)

        # Create a new filename with the .jpg extension
        output_filename = f"{os.path.splitext(filename)[0]}.jpg"
        output_path = os.path.join(output_dir, output_filename)

        try:
            with Image.open(input_path) as img:
                # Resize the image. LANCZOS is a high-quality filter for resizing.
                resized_img = img.resize(target_size, Image.Resampling.LANCZOS)

                # Convert to RGB mode. This is crucial for saving as JPEG,
                # as it removes any alpha (transparency) channel.
                rgb_img = resized_img.convert('RGB')

                # Save the image as a JPEG with the specified quality
                rgb_img.save(output_path, 'jpeg', quality=quality, optimize=True)

        except Exception as e:
            print(f"\n[ERROR] Failed to process {filename}: {e}")

    print("\n[SUCCESS] All images have been processed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script to resize and convert a batch of images to 1280x720 JPG format.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input_dir",
        help="The path to the folder containing your original menu screenshots."
    )
    parser.add_argument(
        "output_dir",
        help="The path to the folder where the processed images will be saved."
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=90,
        help="The JPEG quality for the output images (1-100). Default is 90."
    )

    args = parser.parse_args()

    process_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        quality=args.quality
    )