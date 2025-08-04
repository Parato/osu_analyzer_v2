# chick_tool.py
#
# A simple GUI tool to verify YOLO annotations against their corresponding images.
# MODIFIED: Removed 'spinner' from the class list to match new dataset format.

import os
import random
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk, ImageDraw

# --- Configuration ---
# This should point to the root of your final, assembled dataset.
# The script will look for 'images/train', 'images/val', etc. inside this directory.
DEFAULT_DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "master_dataset_v16")

# MODIFIED: Updated class colors and names to match the new simplified configuration.
# The order must match the dataset's YAML file.
CLASS_COLORS = {
    0: ("#ff0000", "hit_circle"),  # Red
    1: ("#ffff00", "cursor"),      # Yellow
    2: ("#7613f0", "hit_miss")     # Violet
}


class AnnotationChecker:
    """
    A GUI application to load random images from a dataset and display their
    YOLO annotations as colored dots for verification.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Chick Tool - Annotation Verifier")

        # --- Member Variables ---
        self.dataset_dir = DEFAULT_DATASET_DIR
        self.image_files = []
        # --- NEW: Stores a map of basename -> full_path for quick lookups ---
        self.image_path_map = {}
        self.current_index = -1
        self.current_image_path = None
        self.current_annotations = []
        self.original_image = None
        self.display_image = None

        # --- UI Setup ---
        # Frame for top controls
        top_control_frame = tk.Frame(root)
        top_control_frame.pack(pady=5)

        self.browse_button = tk.Button(top_control_frame, text="Browse for Dataset", command=self.browse_for_dataset)
        self.browse_button.pack(side=tk.LEFT, padx=10)

        self.load_button = tk.Button(top_control_frame, text="Load Random Image", command=self.load_random_image)
        self.load_button.pack(side=tk.LEFT, padx=10)

        self.show_annotations_var = tk.BooleanVar(value=True)
        self.show_annotations_check = tk.Checkbutton(
            top_control_frame,
            text="Show Annotations",
            variable=self.show_annotations_var,
            command=self.redraw_canvas
        )
        self.show_annotations_check.pack(side=tk.LEFT, padx=10)

        # --- NEW: Frame for navigation controls ---
        nav_control_frame = tk.Frame(root)
        nav_control_frame.pack(pady=5)

        self.prev_button = tk.Button(nav_control_frame, text="<-- Previous", command=self.load_previous_image)
        self.prev_button.pack(side=tk.LEFT, padx=5)

        self.next_button = tk.Button(nav_control_frame, text="Next -->", command=self.load_next_image)
        self.next_button.pack(side=tk.LEFT, padx=5)

        self.goto_label = tk.Label(nav_control_frame, text="Go to Frame #:")
        self.goto_label.pack(side=tk.LEFT, padx=(15, 0))

        self.goto_entry = tk.Entry(nav_control_frame, width=10)
        self.goto_entry.pack(side=tk.LEFT, padx=5)
        self.goto_entry.bind("<Return>", self.go_to_frame_event) # Bind Enter key

        self.goto_button = tk.Button(nav_control_frame, text="Go", command=self.go_to_frame_by_number)
        self.goto_button.pack(side=tk.LEFT, padx=5)

        # Canvas for the image
        self.canvas = tk.Canvas(root, cursor="cross")
        self.canvas.pack(fill="both", expand=True)

        # Label for filename and legend
        self.info_label = tk.Label(root, text="Load a dataset to begin.", justify=tk.LEFT)
        self.info_label.pack(pady=5)

        # --- Initial Load ---
        self.scan_for_images()
        self.update_legend()

    def browse_for_dataset(self):
        """Opens a dialog to select a new dataset directory."""
        dir_path = filedialog.askdirectory(initialdir=os.path.dirname(DEFAULT_DATASET_DIR))
        if dir_path:
            self.dataset_dir = dir_path
            self.current_index = -1
            self.current_image_path = None
            self.original_image = None
            self.canvas.delete("all")
            self.scan_for_images()

    def scan_for_images(self):
        """Scans the dataset directory for image files and populates the file list."""
        self.image_files = []
        self.image_path_map = {}
        print(f"Scanning for images in: {self.dataset_dir}")
        for subset in ['train', 'val']:
            image_dir = os.path.join(self.dataset_dir, 'images', subset)
            if os.path.isdir(image_dir):
                for filename in os.listdir(image_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        full_path = os.path.join(image_dir, filename)
                        self.image_files.append(full_path)
                        # --- NEW: Populate the path map for quick lookups ---
                        self.image_path_map[os.path.basename(filename)] = full_path

        # --- NEW: Sort the list to make next/prev sequential ---
        self.image_files.sort()

        if self.image_files:
            print(f"Found {len(self.image_files)} images.")
            self.info_label.config(text=f"Found {len(self.image_files)} images. Click 'Load Random Image' to start.")
        else:
            self.info_label.config(
                text=f"Error: No images found in the specified dataset directory:\n{self.dataset_dir}")
            print(f"Warning: No images found in {self.dataset_dir}")

    def load_random_image(self):
        """Selects a random image, loads it and its annotations."""
        if not self.image_files:
            self.info_label.config(text="No images found. Please select a valid dataset directory.")
            return

        # --- NEW: Get a random index and call the loader ---
        self.current_index = random.randint(0, len(self.image_files) - 1)
        self._load_image_by_index(self.current_index)

    # --- NEW: Methods for sequential navigation ---
    def load_next_image(self):
        if self.current_index != -1 and self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self._load_image_by_index(self.current_index)

    def load_previous_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self._load_image_by_index(self.current_index)

    # --- NEW: Methods for jumping to a specific frame ---
    def go_to_frame_event(self, event):
        """Callback for pressing Enter in the entry box."""
        self.go_to_frame_by_number()

    def go_to_frame_by_number(self):
        """Loads an image by its frame number from the entry widget."""
        try:
            frame_num = int(self.goto_entry.get())
        except (ValueError, TypeError):
            messagebox.showerror("Invalid Input", "Please enter a valid frame number.")
            return

        filename_to_find = f"frame_{frame_num:07d}.jpg"
        image_path = self.image_path_map.get(filename_to_find)

        if image_path:
            # Find the index of this path in the sorted list
            try:
                self.current_index = self.image_files.index(image_path)
                self._load_image_by_index(self.current_index)
            except ValueError:
                messagebox.showerror("Error", "Could not find the index for the image file.")
        else:
            messagebox.showinfo("Not Found", f"Frame number {frame_num} ({filename_to_find}) was not found in the dataset.")

    def _load_image_by_index(self, index):
        """The core logic to load an image and its annotations by its list index."""
        if not (0 <= index < len(self.image_files)):
            return

        self.current_image_path = self.image_files[index]

        # Determine the corresponding label path
        label_path = self.current_image_path.replace(os.path.join('images', 'train'), os.path.join('labels', 'train'))
        label_path = label_path.replace(os.path.join('images', 'val'), os.path.join('labels', 'val'))
        base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
        label_path = os.path.join(os.path.dirname(label_path), f"{base_name}.txt")

        # Load the image
        try:
            self.original_image = Image.open(self.current_image_path).convert('RGBA')
        except Exception as e:
            print(f"Error loading image {self.current_image_path}: {e}")
            self.info_label.config(text=f"Error loading image: {os.path.basename(self.current_image_path)}")
            return

        # Load annotations
        self.current_annotations = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    try:
                        parts = line.strip().split()
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:])
                        self.current_annotations.append((class_id, x_center, y_center, width, height))
                    except (ValueError, IndexError) as e:
                        print(f"Could not parse line in {label_path}: {line.strip()} - Error: {e}")
        else:
            print(f"No label file found for {os.path.basename(self.current_image_path)}")

        self.redraw_canvas()
        self.update_legend()

    def redraw_canvas(self):
        """Clears the canvas and redraws the image and annotations."""
        if not self.original_image:
            return

        self.canvas.delete("all")
        img_w, img_h = self.original_image.size

        # Create a drawing layer
        draw_layer = Image.new('RGBA', (img_w, img_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(draw_layer)

        # Draw annotations if the checkbox is checked
        if self.show_annotations_var.get() and self.current_annotations:
            for ann in self.current_annotations:
                class_id, x_center, y_center, width, height = ann

                # Convert normalized coordinates to absolute pixel values
                abs_x_center = x_center * img_w
                abs_y_center = y_center * img_h
                abs_width = width * img_w
                abs_height = height * img_h

                # Calculate top-left (x1, y1) and bottom-right (x2, y2) coordinates
                x1 = abs_x_center - (abs_width / 2)
                y1 = abs_y_center - (abs_height / 2)
                x2 = abs_x_center + (abs_width / 2)
                y2 = abs_y_center + (abs_height / 2)

                color_hex, _ = CLASS_COLORS.get(class_id, ("#FFFFFF", "Unknown"))
                box_line_width = 2

                # Draw the rectangle (bounding box)
                draw.rectangle(
                    (x1, y1, x2, y2),
                    outline=color_hex,
                    width=box_line_width
                )

        # Combine the original image with the drawing layer
        display_img_pil = Image.alpha_composite(self.original_image, draw_layer)
        self.display_image = ImageTk.PhotoImage(display_img_pil)

        # Update canvas
        self.canvas.config(width=img_w, height=img_h)
        self.canvas.create_image(0, 0, anchor="nw", image=self.display_image)

    def update_legend(self):
        """Updates the info label with the current filename and a color legend."""
        filename_text = "No image loaded."
        if self.current_image_path:
            # --- MODIFIED: Show frame index ---
            frame_idx_text = f"Frame: {self.current_index + 1}/{len(self.image_files)}"
            filename_text = f"{frame_idx_text} | ...{os.sep}{os.path.basename(os.path.dirname(self.current_image_path))}{os.sep}{os.path.basename(self.current_image_path)}"

        legend_parts = [f" {CLASS_COLORS[i][1]}" for i in sorted(CLASS_COLORS.keys())]
        legend_text = f"Legend ({len(CLASS_COLORS)} classes):" + " | ".join(legend_parts)

        self.info_label.config(text=f"{filename_text}\n{legend_text}")


def main():
    root = tk.Tk()
    app = AnnotationChecker(root)
    root.mainloop()


if __name__ == "__main__":
    main()