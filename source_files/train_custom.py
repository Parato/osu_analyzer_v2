# train_custom.py
#
# This script trains a YOLOv8 model directly on the video/JSON dataset format
# by implementing a custom PyTorch training loop.
#
# FINAL CORRECTED VERSION 3.0: Implements the definitive fix to prevent the
# unwanted download of the COCO dataset by creating a temporary YAML file and
# overriding the model's data configuration before training begins.

import os
import torch
import argparse
import yaml
from ultralytics import YOLO
from torch.utils.data import Dataset, DataLoader
import cv2
import json
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config_generator as cfg  # Import our config to get the class map


# --- Custom Dataset Class (No changes needed here) ---
class VideoDataset(Dataset):
    """Custom PyTorch Dataset for loading frames from .mp4/.json files."""

    def __init__(self, root_dir, subset='train', transform=None):
        self.subset = subset
        self.transform = transform
        self.video_dir = os.path.join(root_dir, subset, 'videos')
        self.anno_dir = os.path.join(root_dir, subset, 'annotations')
        if not os.path.isdir(self.video_dir) or not os.path.isdir(self.anno_dir):
            raise FileNotFoundError(f"Dataset directory not found for subset '{subset}' in '{root_dir}'")
        self.samples = self._create_sample_list()

    def _create_sample_list(self):
        samples_list = []
        video_files = sorted([f for f in os.listdir(self.video_dir) if f.endswith('.mp4')])
        if not video_files:
            print(f"[WARN] No video files found in the '{self.subset}' set.")
            return []
        print(f"Scanning '{self.subset}' set...")
        for video_file in tqdm(video_files):
            video_path = os.path.join(self.video_dir, video_file)
            base_name = os.path.splitext(video_file)[0]
            json_path = os.path.join(self.anno_dir, f"{base_name}.json")
            if not os.path.exists(json_path): continue
            with open(json_path, 'r') as f:
                annotations = json.load(f)
            video_frames_data = annotations.get('frames', {})
            metadata = annotations.get('metadata', {})
            start_frame_num = int(metadata.get('start_time_ms', 0) * metadata.get('fps', 60) / 1000)
            for frame_key, frame_annos in video_frames_data.items():
                frame_index_in_clip = int(frame_key) - start_frame_num
                samples_list.append((video_path, frame_index_in_clip, frame_annos))
        print(f"Found {len(samples_list)} annotated frames in the '{self.subset}' set.")
        return samples_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, frame_index, annotations = self.samples[idx]
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()
        if not ret: return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bboxes, class_labels = [], []
        for ann in annotations:
            bboxes.append(ann['box'])
            class_labels.append(ann['class_id'])
        if self.transform:
            try:
                transformed = self.transform(image=frame, bboxes=bboxes, class_labels=class_labels)
                frame = transformed['image']
                bboxes = transformed['bboxes']
                class_labels = transformed['class_labels']
            except Exception:
                fallback_transform = A.Compose([A.Resize(640, 640), ToTensorV2()])
                transformed = fallback_transform(image=frame)
                frame = transformed['image']
        targets = torch.zeros((len(bboxes), 5))
        if bboxes:
            targets[:, 0] = torch.tensor(class_labels)
            targets[:, 1:] = torch.tensor(bboxes)
        return frame, targets


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None and x[0] is not None, batch))
    if not batch: return None, None
    images, targets = zip(*batch)
    for i, target in enumerate(targets):
        if target.numel() > 0:
            batch_idx = torch.full((target.shape[0], 1), i)
            targets[i] = torch.cat((batch_idx, target), 1)
    images = torch.stack(images, 0)
    valid_targets = [t for t in targets if t.numel() > 0]
    targets = torch.cat(valid_targets, 0) if valid_targets else torch.tensor([])
    return images, targets


def main():
    parser = argparse.ArgumentParser(description="Custom YOLOv8 Trainer for Video Datasets")
    parser.add_argument("--dataset-dir", required=True, help="Path to the root of the video dataset directory.")
    parser.add_argument("--weights", default="yolov8s.pt", help="Path to initial weights (.pt file).")
    parser.add_argument("--project", default="runs/detect", help="Directory to save training runs.")
    parser.add_argument("--name", default="yolov8s_osu_custom_video", help="Name for the training run directory.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    args = parser.parse_args()

    print("--- Osu! Detector Custom Video Trainer ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    output_dir = os.path.join(args.project, args.name)
    weights_dir = os.path.join(output_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # --- FIX: Prevent COCO download by creating a temporary YAML config ---
    print("Creating temporary dataset YAML to override library defaults...")

    # Get class names in the correct order of their IDs
    class_names = [name for name, id in sorted(cfg.CLASS_MAP.items(), key=lambda item: item[1])]

    dataset_yaml_data = {
        'train': os.path.abspath(os.path.join(args.dataset_dir, 'train')),
        'val': os.path.abspath(os.path.join(args.dataset_dir, 'val')),
        'nc': len(class_names),
        'names': class_names
    }

    temp_yaml_path = os.path.join(output_dir, 'temp_data.yaml')
    with open(temp_yaml_path, 'w') as f:
        yaml.dump(dataset_yaml_data, f, default_flow_style=False)

    print(f"Temporary YAML created at: {temp_yaml_path}")
    # --- END OF FIX ---

    model = YOLO(args.weights).to(device)
    # --- FIX: Override the model's data configuration ---
    model.data = temp_yaml_path

    train_transform = A.Compose([
        A.Resize(640, 640),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    val_transform = A.Compose([
        A.Resize(640, 640),
        ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    train_dataset = VideoDataset(root_dir=args.dataset_dir, subset='train', transform=train_transform)
    val_dataset = VideoDataset(root_dir=args.dataset_dir, subset='val', transform=val_transform)

    if len(train_dataset) == 0:
        print("[ERROR] Training dataset is empty. Aborting training.")
        return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                              collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=0,
                            collate_fn=collate_fn, pin_memory=True)

    optimizer = torch.optim.Adam(model.model.parameters(), lr=args.lr,
                                 weight_decay=0.0005)  # Access internal model for parameters
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_val_loss = float('inf')

    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
        total_train_loss = 0.0

        for images, targets in train_pbar:
            if images is None: continue
            images = images.to(device, non_blocking=True).float() / 255.0
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad()

            # Manually perform forward pass to get loss
            preds = model.model(images)
            loss, _ = model.loss(preds, targets)

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            train_pbar.set_postfix(loss=f"{total_train_loss / (train_pbar.n + 1):.4f}")

        model.eval()
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Val]")
        total_val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_pbar:
                if images is None: continue
                images = images.to(device, non_blocking=True).float() / 255.0
                targets = targets.to(device, non_blocking=True)

                preds = model.model(images)
                loss, _ = model.loss(preds, targets)

                total_val_loss += loss.item()
                val_pbar.set_postfix(loss=f"{total_val_loss / (val_pbar.n + 1):.4f}")

        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        print(
            f"Epoch {epoch + 1} Summary: Avg Train Loss: {total_train_loss / len(train_loader):.4f}, Avg Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(weights_dir, "best.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path} with validation loss: {best_val_loss:.4f}")

        scheduler.step()

    print("\n--- Training Complete ---")
    print(f"The best model has been saved to: {os.path.join(weights_dir, 'best.pt')}")


if __name__ == '__main__':
    main()