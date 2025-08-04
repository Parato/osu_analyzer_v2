# unpacker.py
#
# This script converts our generated video/JSON dataset into the
# standard YOLO format (individual image and label files). This allows
# us to use the standard `train.py` script without modification.

import os
import cv2
import json
import argparse
from tqdm import tqdm


def unpack_dataset(video_dataset_path, output_path):
    """
    Reads a video/JSON dataset and writes it out as an image/label dataset.

    Args:
        video_dataset_path (str): Path to the master video dataset (e.g., 'master_dataset_v16_video').
        output_path (str): Path to save the unpacked, training-ready dataset.
    """
    if not os.path.isdir(video_dataset_path):
        print(f"[ERROR] Input video dataset not found at: {video_dataset_path}")
        return

    print(f"Starting to unpack dataset from '{video_dataset_path}' to '{output_path}'...")

    class_map = {}

    # Process both 'train' and 'val' subsets
    for subset in ['train', 'val']:
        print(f"\nProcessing '{subset}' subset...")

        video_dir = os.path.join(video_dataset_path, subset, 'videos')
        anno_dir = os.path.join(video_dataset_path, subset, 'annotations')

        if not os.path.isdir(video_dir) or not os.path.isdir(anno_dir):
            print(f"[WARN] Could not find 'videos' or 'annotations' directory for subset '{subset}'. Skipping.")
            continue

        # Create the destination directories
        output_img_dir = os.path.join(output_path, 'images', subset)
        output_lbl_dir = os.path.join(output_path, 'labels', subset)
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_lbl_dir, exist_ok=True)

        video_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])

        frame_counter = 0

        for video_file in tqdm(video_files, desc=f"Unpacking {subset} videos"):
            video_path = os.path.join(video_dir, video_file)
            base_name = os.path.splitext(video_file)[0]
            json_path = os.path.join(anno_dir, f"{base_name}.json")

            if not os.path.exists(json_path):
                continue

            # Load the entire annotation file for this video
            with open(json_path, 'r') as f:
                annotations = json.load(f)

            # Grab the class map from the first valid JSON we find
            if not class_map and 'metadata' in annotations and 'class_map' in annotations['metadata']:
                class_map = annotations['metadata']['class_map']

            video_frames_data = annotations.get('frames', {})
            start_frame_num = int(
                annotations.get('metadata', {}).get('start_time_ms', 0) * annotations.get('metadata', {}).get('fps',
                                                                                                              60) / 1000)

            # Open the video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"[WARN] Could not open video {video_file}. Skipping.")
                continue

            # Extract every frame that has an annotation
            for frame_key, frame_annos in video_frames_data.items():
                frame_index_in_clip = int(frame_key) - start_frame_num

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index_in_clip)
                ret, frame = cap.read()

                if ret:
                    # Save the image
                    image_filename = f"{base_name}_frame_{frame_index_in_clip}.jpg"
                    cv2.imwrite(os.path.join(output_img_dir, image_filename), frame)

                    # Save the corresponding label file
                    label_filename = f"{base_name}_frame_{frame_index_in_clip}.txt"
                    with open(os.path.join(output_lbl_dir, label_filename), 'w') as lbl_file:
                        for ann in frame_annos:
                            class_id = ann['class_id']
                            box_str = " ".join(map(str, ann['box']))
                            lbl_file.write(f"{class_id} {box_str}\n")

            cap.release()

    # --- Final Step: Create the dataset.yaml file ---
    if class_map:
        print("\nCreating dataset YAML file...")
        yaml_path = os.path.join(output_path, "dataset.yaml")

        # Invert class map to get names in order
        names_list = [name for id, name in sorted(class_map.items(), key=lambda item: item[1])]

        with open(yaml_path, 'w') as f:
            f.write(f"train: {os.path.abspath(os.path.join(output_path, 'images/train'))}\n")
            f.write(f"val: {os.path.abspath(os.path.join(output_path, 'images/val'))}\n")
            f.write(f"\n")
            f.write(f"# number of classes\n")
            f.write(f"nc: {len(names_list)}\n")
            f.write(f"\n")
            f.write(f"# class names\n")
            f.write(f"names: {names_list}\n")

        print(f"Successfully created '{yaml_path}'")
    else:
        print("[WARN] Could not determine class map. The 'dataset.yaml' file was not created.")

    print("\n--- Unpacking Complete! ---")
    print(f"The training-ready dataset is located at: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Unpacks a video/JSON dataset into a standard image/label format for YOLO training.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "video_dataset_path",
        help="Path to the input video dataset directory (e.g., 'datasets/master_dataset_v16_video')."
    )
    parser.add_argument(
        "output_path",
        help="Path to the output directory for the unpacked dataset (e.g., 'datasets/master_dataset_v16_unpacked')."
    )
    args = parser.parse_args()

    unpack_dataset(args.video_dataset_path, args.output_path)


if __name__ == "__main__":
    main()