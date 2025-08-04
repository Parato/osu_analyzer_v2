# annotator.py
#
# A simple video annotation tool to help create a dataset for training the
# YOLO object detection model.
# MODIFIED: Removed 'spinner' class.

import cv2
import argparse
import os

# Get the absolute path of the directory where this script is located.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Configuration ---
# These should match the classes you intend to train your model on.
# The order here determines the class ID (0, 1, 2, ...).
CLASSES = ["hit_circle", "cursor", "hit_miss"]
# UPDATED: Output directory is now correctly placed inside the source_files folder.
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "annotations")

# --- Global State ---
drawing = False
ref_point = []
current_frame_boxes = []
selected_class_id = 0


def draw_bounding_box(event, x, y, flags, param):
    """Mouse callback function for drawing bounding boxes."""
    global ref_point, drawing, current_frame_boxes, selected_class_id

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        drawing = True

    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        drawing = False
        # Add the completed box to the list for this frame
        current_frame_boxes.append({
            "box": (ref_point[0][0], ref_point[0][1], x - ref_point[0][0], y - ref_point[0][1]),
            "class_id": selected_class_id
        })


def save_annotations(frame_number, frame_width, frame_height, video_name):
    """Saves the bounding boxes for the current frame to a text file in YOLO format."""
    if not current_frame_boxes:
        print(f"Frame {frame_number}: No boxes to save.")
        return

    # This will now create the directory at the correct path if it doesn't exist.
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # YOLO format: <class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
    file_path = os.path.join(OUTPUT_DIR, f"{video_name}_frame_{frame_number}.txt")

    with open(file_path, 'w') as f:
        for item in current_frame_boxes:
            box = item['box']
            class_id = item['class_id']

            x, y, w, h = box

            # Normalize coordinates
            x_center_norm = (x + w / 2) / frame_width
            y_center_norm = (y + h / 2) / frame_height
            width_norm = w / frame_width
            height_norm = h / frame_height

            f.write(f"{class_id} {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n")

    print(f"Frame {frame_number}: Saved {len(current_frame_boxes)} annotations to {file_path}")


def main():
    global current_frame_boxes, selected_class_id

    parser = argparse.ArgumentParser(description="A tool for annotating osu! gameplay videos.")
    parser.add_argument("video_path", help="The full path to the gameplay video file.")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {args.video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_number = 0

    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    window_name = "Osu! Annotator"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_bounding_box)

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        clone = frame.copy()

        # Display current boxes
        for item in current_frame_boxes:
            x, y, w, h = item['box']
            cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(clone, CLASSES[item['class_id']], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display UI info
        info_text = f"Frame: {frame_number}/{total_frames} | Class: {CLASSES[selected_class_id].upper()}"
        cv2.putText(clone, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow(window_name, clone)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('d'):  # Next frame
            frame_number = min(frame_number + 1, total_frames - 1)
            current_frame_boxes = []  # Clear boxes for new frame
        elif key == ord('a'):  # Previous frame
            frame_number = max(frame_number - 1, 0)
            current_frame_boxes = []  # Clear boxes for new frame
        elif key == ord('s'):  # Save
            save_annotations(frame_number, frame_width, frame_height, video_name)
        elif ord('0') <= key <= ord('9'):  # Select class
            class_id = key - ord('0')
            if class_id < len(CLASSES):
                selected_class_id = class_id
                print(f"Selected class: {CLASSES[selected_class_id]}")
            else:
                print(f"Invalid class ID: {class_id}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()