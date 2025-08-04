# data_loader_stub.py
#
# This script serves as a template or "stub" for creating a custom PyTorch
# Dataset to read from our new video-based dataset format (.mp4 + .json).
# It demonstrates the core logic required to feed the data into a model for training.

import os
import cv2
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


class VideoDataset(Dataset):
    """
    A custom PyTorch Dataset for loading video frames and their corresponding
    annotations from .mp4 and .json files.
    """

    def __init__(self, root_dir, subset='train', transform=None):
        """
        Args:
            root_dir (str): The root directory of the master dataset (e.g., 'master_dataset_v16_video').
            subset (str): The dataset subset to load ('train' or 'val').
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        self.video_dir = os.path.join(root_dir, subset, 'videos')
        self.anno_dir = os.path.join(root_dir, subset, 'annotations')

        self.video_files = sorted([f for f in os.listdir(self.video_dir) if f.endswith('.mp4')])

        # --- This is the core data structure ---
        # It's a list of tuples, where each tuple contains:
        # (video_path, json_data, frame_key_in_json)
        self.samples = self._create_sample_list()

    def _create_sample_list(self):
        """
        Parses all JSON files to create a flat list of every frame that has an annotation.
        This is more efficient than opening and closing files for every getItem call.
        """
        print(f"Loading '{self.subset}' set samples...")
        samples_list = []
        for video_file in self.video_files:
            video_path = os.path.join(self.video_dir, video_file)
            base_name = os.path.splitext(video_file)[0]
            json_path = os.path.join(self.anno_dir, f"{base_name}.json")

            if not os.path.exists(json_path):
                print(f"[WARN] Annotation file not found for video {video_file}, skipping.")
                continue

            with open(json_path, 'r') as f:
                annotations = json.load(f)

            # The 'frames' dict in our JSON contains frame numbers as keys
            # and a list of annotation objects as values.
            video_frames = annotations.get('frames', {})

            # The original frame number when the clip was generated
            start_frame_num = int(
                annotations.get('metadata', {}).get('start_time_ms', 0) * annotations.get('metadata', {}).get('fps',
                                                                                                              60) / 1000)

            for frame_key, frame_annos in video_frames.items():
                # frame_key is the original frame number string, e.g., "12345"
                # We need the 0-indexed position within this specific clip
                frame_index_in_clip = int(frame_key) - start_frame_num
                samples_list.append((video_path, frame_index_in_clip, frame_annos))

        print(f"Found {len(samples_list)} annotated frames in {len(self.video_files)} video clips.")
        return samples_list

    def __len__(self):
        """Returns the total number of annotated frames."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Fetches a single sample (frame and its annotations) from the dataset.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video_path, frame_index, annotations = self.samples[idx]

        # Use a persistent VideoCapture object if performance is critical,
        # but opening per-item is safer and simpler for a stub.
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            # Handle the case where the frame could not be read
            print(f"[ERROR] Could not read frame {frame_index} from {video_path}")
            # Return a dummy sample or raise an error
            return torch.zeros((3, 640, 480)), torch.zeros((0, 5))

        # Convert frame from BGR to RGB and to a tensor
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Prepare targets in the format [class_id, x_center, y_center, width, height]
        targets = []
        for ann in annotations:
            class_id = ann['class_id']
            box = ann['box']  # [x_center, y_center, width, height]
            targets.append([class_id] + box)

        targets = torch.tensor(targets, dtype=torch.float32)

        # Apply transformations if any (e.g., for data augmentation)
        if self.transform:
            # Note: Albumentations is a popular library for this
            transformed = self.transform(image=frame, bboxes=targets[:, 1:], class_labels=targets[:, 0])
            frame = transformed['image']

            # Reconstruct targets if augmentations changed them
            new_targets = torch.cat([
                transformed['class_labels'].unsqueeze(1),
                transformed['bboxes']
            ], dim=1)
            targets = new_targets

        # Convert frame to tensor format (C, H, W)
        frame_tensor = torch.from_numpy(frame.transpose((2, 0, 1))).float() / 255.0

        return frame_tensor, targets


# --- Example Usage ---
def main():
    """Demonstrates how to use the VideoDataset class."""

    # Path to the dataset we generated with master_pipeline.py
    dataset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "master_dataset_v16_video")

    if not os.path.exists(dataset_root):
        print(f"Error: Dataset not found at '{dataset_root}'")
        print("Please run the master_pipeline.py script first to generate the dataset.")
        return

    # Create an instance of the dataset for the training set
    print("Initializing the training dataset...")
    train_dataset = VideoDataset(root_dir=dataset_root, subset='train')

    # You can add transforms here for augmentation using libraries like Albumentations

    # Create a DataLoader to handle batching, shuffling, etc.
    # This is what you would feed to your training loop.
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0,
                                  collate_fn=lambda x: tuple(zip(*x)))  # Custom collate for YOLO format

    print("\n--- Example Batch ---")
    # Get one batch of data to see the output
    try:
        images, targets = next(iter(train_dataloader))

        print(f"Number of images in batch: {len(images)}")
        print(f"Shape of the first image tensor: {images[0].shape}")

        print("\nAnnotations for the first image in the batch:")
        first_target = targets[0]
        print(f"Number of objects: {first_target.shape[0]}")
        print("Format: [class_id, x_center, y_center, width, height]")
        print(first_target)

    except StopIteration:
        print("Could not retrieve a batch. The dataset might be empty or there was an issue loading.")
    except Exception as e:
        print(f"An error occurred while loading a batch: {e}")


if __name__ == "__main__":
    main()