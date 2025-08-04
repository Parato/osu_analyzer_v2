# video_player_tool.py
#
# A GUI tool to play generated .mp4 clips and overlay their corresponding
# .json annotations for verification. This replaces the old `chick_tool.py`.

import os
import cv2
import json
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk, ImageDraw

# --- Configuration ---
# Default directory to look for datasets
DEFAULT_DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")

# BGR colors for drawing bounding boxes
CLASS_COLORS = {
    0: ("#ff0000", "hit_circle"),  # Red
    1: ("#ffff00", "cursor"),      # Yellow
    2: ("#ffffff", "spinner"),     # White
    3: ("#7613f0", "hit_miss")      # Violet
}


class VideoPlayer:
    """
    A GUI application to play a video file and its corresponding JSON annotations.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Video Annotation Player")

        # --- Member Variables ---
        self.cap = None
        self.video_path = None
        self.annotations = {}
        self.is_playing = False
        self.frame_count = 0
        self.fps = 30

        # --- UI Setup ---
        # Top frame for controls
        control_frame = tk.Frame(root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        self.load_button = tk.Button(control_frame, text="Load Video Clip", command=self.load_video)
        self.load_button.pack(side=tk.LEFT, padx=5)

        self.play_pause_button = tk.Button(control_frame, text="Play", command=self.toggle_play_pause, state=tk.DISABLED)
        self.play_pause_button.pack(side=tk.LEFT, padx=5)

        self.info_label = tk.Label(control_frame, text="No video loaded.")
        self.info_label.pack(side=tk.LEFT, padx=10)

        # Canvas for the video
        self.canvas = tk.Canvas(root, bg="black")
        self.canvas.pack(fill="both", expand=True, padx=10, pady=5)

        # Bottom frame for slider
        slider_frame = tk.Frame(root)
        slider_frame.pack(fill=tk.X, padx=10, pady=5)

        self.slider_label = tk.Label(slider_frame, text="Frame: 0/0")
        self.slider_label.pack(side=tk.LEFT)

        self.slider = ttk.Scale(slider_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_slider_move, state=tk.DISABLED)
        self.slider.pack(fill=tk.X, expand=True, padx=10)

    def load_video(self):
        """Opens a file dialog to select a video file and its annotations."""
        initial_dir = os.path.join(DEFAULT_DATASET_DIR)
        video_path = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="Select a video file",
            filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*"))
        )
        if not video_path:
            return

        json_path = os.path.splitext(video_path)[0] + '.json'
        if not os.path.exists(json_path):
            self.info_label.config(text=f"Error: Annotation file not found at {os.path.basename(json_path)}")
            return

        self.video_path = video_path
        self.is_playing = False
        self.play_pause_button.config(text="Play")

        # Load video with OpenCV
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.video_path)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        # Load annotations
        with open(json_path, 'r') as f:
            full_json = json.load(f)
            self.annotations = full_json.get('frames', {})
            # Use original frame numbers from metadata if available, otherwise assume 0-indexed
            self.start_frame = full_json.get('metadata', {}).get('start_time_ms', 0) * self.fps / 1000

        # Update UI
        self.play_pause_button.config(state=tk.NORMAL)
        self.slider.config(state=tk.NORMAL, to=self.frame_count - 1)
        self.slider.set(0)
        self.info_label.config(text=f"Loaded: {os.path.basename(self.video_path)}")

        self.show_frame(0)

    def toggle_play_pause(self):
        """Switches between playing and pausing the video."""
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_pause_button.config(text="Pause")
            self.play_video()
        else:
            self.play_pause_button.config(text="Play")

    def play_video(self):
        """The main playback loop."""
        if not self.is_playing or not self.cap:
            return

        current_frame_pos = int(self.slider.get())
        if current_frame_pos < self.frame_count - 1:
            next_frame_pos = current_frame_pos + 1
            self.show_frame(next_frame_pos)
            self.slider.set(next_frame_pos)
            # Schedule the next frame
            self.root.after(int(1000 / self.fps), self.play_video)
        else:
            # End of video
            self.is_playing = False
            self.play_pause_button.config(text="Play")

    def on_slider_move(self, value):
        """Callback for when the user moves the slider."""
        if self.cap:
            self.show_frame(int(float(value)))

    def show_frame(self, frame_index):
        """Reads a specific frame, draws annotations, and displays it."""
        if not self.cap or frame_index >= self.frame_count:
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        if not ret:
            return

        # Convert from BGR (cv2) to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = frame.shape

        pil_img = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_img)

        # Check for annotations for this frame
        # The JSON keys are strings, so we must convert the index
        original_frame_number = str(int(self.start_frame + frame_index))
        if original_frame_number in self.annotations:
            for ann in self.annotations[original_frame_number]:
                class_id = ann['class_id']
                x_center, y_center, width, height = ann['box']

                # Convert normalized YOLO format to pixel coordinates
                abs_x = x_center * img_w
                abs_y = y_center * img_h
                abs_w = width * img_w
                abs_h = height * img_h

                x1 = abs_x - abs_w / 2
                y1 = abs_y - abs_h / 2
                x2 = abs_x + abs_w / 2
                y2 = abs_y + abs_h / 2

                color_hex, label_name = CLASS_COLORS.get(class_id, ("#FFFFFF", "Unknown"))
                draw.rectangle([x1, y1, x2, y2], outline=color_hex, width=2)
                draw.text((x1, y1 - 12), label_name, fill=color_hex)

        # Display on canvas
        self.photo = ImageTk.PhotoImage(image=pil_img)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.slider_label.config(text=f"Frame: {frame_index}/{self.frame_count - 1}")
        self.root.update_idletasks() # Ensure UI updates


def main():
    root = tk.Tk()
    app = VideoPlayer(root)
    root.geometry("1024x768")
    root.mainloop()


if __name__ == "__main__":
    main()