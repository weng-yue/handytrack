"""
ManualTifTracker with Integrated Particle Detection
Combines the manual tracking interface with automatic particle detection visualization.
Shows detected particle centers on each frame while maintaining all manual tracking functionality.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json


IMAGE_EXTENSIONS = [".tif"]

# Visualization parameters
MARKER_RADIUS = 5
MARKER_WIDTH = 2
HISTORY_RADIUS = 2
HISTORY_LINE_WIDTH = 2


# ============= PARTICLE DETECTOR (Complete Original Code) =============

@dataclass
class Particle:
    """Data class for detected particles"""
    id: int
    centroid_x: float
    centroid_y: float
    area: int
    bounding_box: Tuple[int, int, int, int]  # (x, y, width, height)
    density: float
    confidence: float


class ParticleDetector:
    """
    Detects particles in binary images using connected components analysis.
    """

    def __init__(self,
                 min_area: int = 50,
                 max_area: int = 5000,
                 min_density: float = 0.3,
                 threshold_value: int = 128,
                 use_morphology: bool = True):
        """
        Initialize the particle detector.

        Args:
            min_area: Minimum pixel area for a valid particle
            max_area: Maximum pixel area for a valid particle
            min_density: Minimum white pixel density (0-1) in bounding box
            threshold_value: Threshold for converting to binary (if not already binary)
            use_morphology: Apply morphological operations to reduce noise
        """
        self.min_area = min_area
        self.max_area = max_area
        self.min_density = min_density
        self.threshold_value = threshold_value
        self.use_morphology = use_morphology

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image: convert to binary and optionally denoise.

        Args:
            image: Input grayscale image (numpy array)

        Returns:
            Binary image (0 or 255)
        """
        # Ensure grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply threshold if not already binary
        _, binary = cv2.threshold(image, self.threshold_value, 255, cv2.THRESH_BINARY)

        # Optional: morphological operations to remove noise
        if self.use_morphology:
            # Opening: removes small white noise
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

            # Closing: fills small holes in particles
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        return binary

    def calculate_density(self, binary_image: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """
        Calculate white pixel density within bounding box.

        Args:
            binary_image: Binary image
            bbox: Bounding box (x, y, width, height)

        Returns:
            Density ratio (0-1)
        """
        x, y, w, h = bbox
        roi = binary_image[y:y+h, x:x+w]
        white_pixels = np.sum(roi == 255)
        total_pixels = w * h
        return white_pixels / total_pixels if total_pixels > 0 else 0

    def calculate_confidence(self, area: int, density: float) -> float:
        """
        Calculate confidence score for a particle (0-1).
        Based on how well it matches expected characteristics.

        Args:
            area: Particle area in pixels
            density: Pixel density in bounding box

        Returns:
            Confidence score (0-1)
        """
        # Normalize area score (closer to middle of range = higher score)
        area_range = self.max_area - self.min_area
        area_middle = (self.max_area + self.min_area) / 2
        area_score = 1.0 - abs(area - area_middle) / (area_range / 2)
        area_score = max(0, min(1, area_score))

        # Density score (higher is better)
        density_score = min(1.0, density / 0.8)  # 0.8 is "ideal" density

        # Combined confidence (weighted average)
        confidence = 0.6 * density_score + 0.4 * area_score
        return round(confidence, 3)

    def detect_particles(self, image_path: str) -> List[Particle]:
        """
        Detect particles in an image file.

        Args:
            image_path: Path to image file (TIF, PNG, etc.)

        Returns:
            List of detected Particle objects, sorted by confidence (highest first)
        """
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Preprocess
        binary = self.preprocess_image(image)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        particles = []

        # Iterate through components (skip label 0 which is background)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]

            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue

            # Extract bounding box
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            bbox = (x, y, w, h)

            # Calculate density
            density = self.calculate_density(binary, bbox)

            # Filter by density
            if density < self.min_density:
                continue

            # Get centroid
            cx, cy = centroids[i]

            # Calculate confidence
            confidence = self.calculate_confidence(area, density)

            # Create particle object
            particle = Particle(
                id=i,
                centroid_x=float(cx),
                centroid_y=float(cy),
                area=int(area),
                bounding_box=bbox,
                density=float(density),
                confidence=confidence
            )
            particles.append(particle)

        # Sort by confidence (highest first)
        particles.sort(key=lambda p: p.confidence, reverse=True)

        return particles

    def detect_particles_from_array(self, image: np.ndarray) -> List[Particle]:
        """
        Detect particles in an image array (for integration with tracker).

        Args:
            image: Grayscale image as numpy array

        Returns:
            List of detected Particle objects, sorted by confidence (highest first)
        """
        if image is None:
            return []

        # Preprocess
        binary = self.preprocess_image(image)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        particles = []

        # Iterate through components (skip label 0 which is background)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]

            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue

            # Extract bounding box
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            bbox = (x, y, w, h)

            # Calculate density
            density = self.calculate_density(binary, bbox)

            # Filter by density
            if density < self.min_density:
                continue

            # Get centroid
            cx, cy = centroids[i]

            # Calculate confidence
            confidence = self.calculate_confidence(area, density)

            # Create particle object
            particle = Particle(
                id=i,
                centroid_x=float(cx),
                centroid_y=float(cy),
                area=int(area),
                bounding_box=bbox,
                density=float(density),
                confidence=confidence
            )
            particles.append(particle)

        # Sort by confidence (highest first)
        particles.sort(key=lambda p: p.confidence, reverse=True)

        return particles

    def detect_batch(self, folder_path: str, output_folder: Optional[str] = None) -> dict:
        """
        Detect particles in all images in a folder.

        Args:
            folder_path: Path to folder containing images
            output_folder: Optional path to save detection JSON files

        Returns:
            Dictionary mapping filenames to particle lists
        """
        results = {}

        # Get all image files
        image_files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg'))
        ])

        for filename in image_files:
            image_path = os.path.join(folder_path, filename)

            try:
                particles = self.detect_particles(image_path)
                results[filename] = particles

                # Optionally save to JSON
                if output_folder:
                    os.makedirs(output_folder, exist_ok=True)
                    json_path = os.path.join(
                        output_folder,
                        filename.replace(os.path.splitext(filename)[1], '.json')
                    )
                    self.save_detections(particles, json_path)

                print(f"✓ {filename}: {len(particles)} particles detected")

            except Exception as e:
                print(f"✗ {filename}: Error - {str(e)}")
                results[filename] = []

        return results

    @staticmethod
    def save_detections(particles: List[Particle], output_path: str):
        """
        Save detected particles to JSON file.

        Args:
            particles: List of Particle objects
            output_path: Path to output JSON file
        """
        data = {
            "num_particles": len(particles),
            "particles": [
                {
                    "id": p.id,
                    "centroid": {"x": p.centroid_x, "y": p.centroid_y},
                    "area": p.area,
                    "bounding_box": {"x": p.bounding_box[0], "y": p.bounding_box[1],
                                     "width": p.bounding_box[2], "height": p.bounding_box[3]},
                    "density": p.density,
                    "confidence": p.confidence
                }
                for p in particles
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def visualize_detections(image_path: str, particles: List[Particle],
                            output_path: Optional[str] = None, show: bool = True):
        """
        Visualize detected particles on the image.

        Args:
            image_path: Path to original image
            particles: List of detected particles
            output_path: Optional path to save visualization
            show: Whether to display image window
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Convert to RGB for proper color display
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Draw particles
        for particle in particles:
            cx, cy = int(particle.centroid_x), int(particle.centroid_y)
            x, y, w, h = particle.bounding_box

            # Color based on confidence (green = high, yellow = medium, red = low)
            if particle.confidence > 0.7:
                color = (0, 255, 0)  # Green
            elif particle.confidence > 0.5:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red

            # Draw bounding box
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)

            # Draw centroid
            cv2.circle(image, (cx, cy), 5, color, -1)
            cv2.circle(image, (cx, cy), 8, color, 2)

            # Add label
            label = f"#{particle.id} ({particle.confidence:.2f})"
            cv2.putText(image, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, color, 1, cv2.LINE_AA)

        # Save if requested
        if output_path:
            cv2.imwrite(output_path, image)

        # Display if requested
        if show:
            cv2.imshow('Particle Detections', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return image


# ============= MANUAL TRACKER (Complete Original Code with Detection Integration) =============

class ManualTifTracker:
    """
    ManualTifTracker: Main application class for manual tracking of .tif image sequences.
    Now includes particle detection visualization overlay.

    Attributes:
    -----------
    root : tk.Tk
        The main Tkinter root window.
    pic_folder : str
        Path to the folder containing .tif images.
    track_folder : str
        Path to the folder where tracking CSV files are saved.
    style : ttk.Style
        Tkinter style object for theming.
    current_theme : str
        Current theme ("dark" or "light").
    tif_paths : list
        List of paths to .tif images.
    current_index : int
        Index of the currently displayed frame.
    tracks : list
        List of dicts representing the current track.
    csv_path : str
        Path to the last saved CSV file.
    threshold_enabled : bool
        Whether thresholding is enabled for image display.
    show_saved_flag : bool
        Whether to show saved tracks overlay.
    show_saved_history : tk.BooleanVar
        Whether to show history of saved tracks.
    tk_image : ImageTk.PhotoImage
        Current image displayed on the canvas.
    display_scale : float
        Scale factor for displaying images on the canvas.
    particle_detector : ParticleDetector
        Instance of particle detector for automatic detection.
    show_detections : tk.BooleanVar
        Whether to show detected particle centers.

    Methods:
    --------
    set_theme(theme)
        Set the UI theme (dark or light).
    toggle_theme()
        Toggle between dark and light themes.
    show_settings_window()
        Display a window for changing key bindings.
    show_detector_settings()
        Display a window for adjusting particle detector parameters.
    validate_number(value_if_allowed)
        Validate numeric input for frame jumping.
    update_jump_entry()
        Update the jump-to-frame entry box.
    toggle_threshold()
        Enable or disable thresholding for image display.
    toggle_detections()
        Toggle display of detected particle centers.
    load_tif_folder(folder)
        Load .tif images from the specified folder.
    show_frame()
        Display the current frame and overlays on the canvas.
    show_detected_particles()
        Overlay detected particle centers on current frame.
    on_click(event)
        Handle mouse click events for adding track points.
    on_mouse_move(event)
        Update coordinate label based on mouse position.
    on_slider_move(value)
        Handle slider movement for frame navigation.
    goto_next_frame()
        Advance to the next frame.
    goto_prev_frame()
        Go back to the previous frame.
    on_mouse_scroll(event)
        Handle mouse wheel scrolling for frame navigation (Windows/Mac).
    on_mouse_scroll_linux(event)
        Handle mouse wheel scrolling for frame navigation (Linux).
    save_csv()
        Save the current track to a CSV file.
    toggle_show_saved_tracks()
        Toggle the display of saved tracks overlay.
    show_saved_tracks()
        Overlay saved tracks from CSV files on the current frame.
    start_new_track()
        Clear the current track and start a new one.
    update_total_tracks_label()
        Update the label showing the total number of saved tracks.
    jump_to_frame()
        Jump to a specific frame based on user input.
    """

    def __init__(self, root, pic_folder, track_folder):
        self.root = root
        self.root.title("Manual Tracker with Particle Detection")
        self.root.geometry("900x900")

        # Store user-selected folders
        self.pic_folder = pic_folder
        self.track_folder = track_folder

        # ---- THEME STYLING ----
        self.style = ttk.Style()
        self.default_font = ("Helvetica", 11)
        self.current_theme = "dark"
        self.set_theme(self.current_theme)

        # Init vars BEFORE loading images
        self.show_saved_flag = False
        self.show_saved_history = tk.BooleanVar(value=False)
        self.show_detections = tk.BooleanVar(value=False)
        self.tif_paths = []
        self.current_index = 0
        self.tracks = []
        self.csv_path = None
        self.tk_image = None
        self.threshold_enabled = True  # Changed to True by default
        self.selecting_particle = False  # Flag for particle selection mode
        self.selected_particle_pos = None  # Store selected particle position

        # Initialize particle detector with default parameters
        self.particle_detector = ParticleDetector(
            min_area=50,
            max_area=5000,
            min_density=0.3,
            threshold_value=0,  # Changed to 0 by default
            use_morphology=True
        )
        
        # Enhanced tracking parameters
        self.tracking_history = []  # Store recent positions for velocity estimation
        self.max_tracking_history = 10  # Keep last 10 positions

        # First control row
        control_row1 = ttk.Frame(self.root)
        control_row1.pack(fill="x", side="top", pady=3)
        ttk.Button(control_row1, text="New Track (n)", command=self.start_new_track).pack(side="left", padx=5)
        ttk.Button(control_row1, text="Save CSV (s)", command=self.save_csv).pack(side="left", padx=5)
        self.toggle_thresh_button = ttk.Button(control_row1, text="Threshold (t)", command=self.toggle_threshold)
        self.toggle_thresh_button.pack(side="left", padx=5)
        ttk.Button(control_row1, text="Settings", command=self.show_settings_window).pack(side="left", padx=5)
        ttk.Button(control_row1, text="Detector Settings", command=self.show_detector_settings).pack(side="left", padx=5)
        ttk.Label(control_row1, text="Skip Gap:").pack(side="left", padx=5)
        self.gap_var = tk.IntVar(value=1)
        spin = tk.Spinbox(control_row1, from_=1, to=100, textvariable=self.gap_var, width=5, state="readonly",
                          font=self.default_font, justify="center")
        spin.pack(side="left")
        self.track_len_label = ttk.Label(control_row1, text="Track Length: 0")
        self.track_len_label.pack(side="left", padx=10)

        # --- Light/Dark mode switch ---
        self.mode_var = tk.StringVar(value="Dark")
        self.mode_switch = ttk.Checkbutton(control_row1, text="Dark Mode", variable=self.mode_var,
                                           onvalue="Dark", offvalue="Light", command=self.toggle_theme)
        self.mode_switch.pack(side="right", padx=5)

        # Second control row
        control_row2 = ttk.Frame(self.root)
        control_row2.pack(fill="x", side="top", pady=3)
        ttk.Checkbutton(control_row2, text="Show Track History", variable=self.show_saved_history,
                        command=self.show_frame).pack(side="left", padx=5)
        ttk.Checkbutton(control_row2, text="Show Detections (d)", variable=self.show_detections,
                        command=self.toggle_detections).pack(side="left", padx=5)
        ttk.Button(control_row2, text="Auto Track Frame (a)", command=self.auto_track_current_frame).pack(side="left", padx=5)
        ttk.Button(control_row2, text="Auto Track All (Shift+A)", command=self.auto_track_until_lost).pack(side="left", padx=5)
        ttk.Button(control_row2, text="Select & Track (c)", command=self.select_and_track_particle).pack(side="left", padx=5)
        self.show_saved_button = ttk.Button(control_row2, text="Show Saved Tracks (h)", command=self.toggle_show_saved_tracks)
        self.show_saved_button.pack(side="left", padx=5)
        self.frame_label = ttk.Label(control_row2, text="Frame: 0")
        self.frame_label.pack(side="left", padx=10)
        self.detection_label = ttk.Label(control_row2, text="Detected: 0")
        self.detection_label.pack(side="left", padx=10)
        self.coord_label = ttk.Label(control_row2, text="X: -, Y: -")
        self.coord_label.pack(side="right", padx=10)
        self.total_tracks_label = ttk.Label(control_row2, text="Total Tracks: 0")
        self.total_tracks_label.pack(side="right", padx=10)

        self.canvas_frame = ttk.Frame(self.root)
        self.canvas_frame.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(self.canvas_frame, cursor="cross", bg="black")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scroll_x = tk.Scrollbar(self.canvas_frame, orient="horizontal", command=self.canvas.xview)
        self.scroll_y = tk.Scrollbar(self.canvas_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.scroll_x.set, yscrollcommand=self.scroll_y.set)
        self.scroll_x.pack(side="bottom", fill="x")
        self.scroll_y.pack(side="right", fill="y")

        self.slider = tk.Scale(self.root, from_=1, to=1, orient="horizontal",
                               command=self.on_slider_move, label="Jump to Frame",
                               font=self.default_font, bg="#2E2E2E", fg="white", troughcolor="#007BFF")
        self.slider.pack(fill="x", pady=3)

        # Frame jump input
        jump_frame_row = ttk.Frame(self.root)
        jump_frame_row.pack(fill="x", pady=5)
        ttk.Label(jump_frame_row, text="Go to Frame:").pack(side="left", padx=5)

        self.jump_var = tk.StringVar()
        self.jump_entry = ttk.Entry(jump_frame_row, width=6, textvariable=self.jump_var)
        self.jump_entry.pack(side="left", padx=5)

        self.jump_entry['validatecommand'] = (self.jump_entry.register(self.validate_number), '%P')
        ttk.Button(jump_frame_row, text="Go", command=self.jump_to_frame).pack(side="left", padx=5)

        self.threshold_slider = tk.Scale(
            self.root, from_=0, to=255, resolution=1,
            orient="horizontal", label="Threshold",
            command=lambda val: self.show_frame(),
            font=self.default_font, bg="#2E2E2E", fg="white", troughcolor="#007BFF"
        )
        self.threshold_slider.set(0)  # Changed to 0 by default
        self.threshold_slider.pack(fill="x", pady=3)

        # Events
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.root.bind("<period>", lambda e: self.goto_next_frame())
        self.root.bind("<comma>", lambda e: self.goto_prev_frame())
        self.root.bind("<Return>", lambda e: self.save_csv())
        self.canvas.bind("<MouseWheel>", self.on_mouse_scroll)
        self.canvas.bind("<Button-4>", self.on_mouse_scroll_linux)
        self.canvas.bind("<Button-5>", self.on_mouse_scroll_linux)

        # --- Keyboard shortcuts ---
        self.root.bind("n", lambda e: self.start_new_track())
        self.root.bind("s", lambda e: self.save_csv())
        self.root.bind("t", lambda e: self.toggle_threshold())
        self.root.bind("h", lambda e: self.toggle_show_saved_tracks())
        self.root.bind("d", lambda e: self.toggle_detections())
        self.root.bind("a", lambda e: self.auto_track_current_frame())
        self.root.bind("A", lambda e: self.auto_track_until_lost())  # Shift+A
        self.root.bind("c", lambda e: self.select_and_track_particle())

        self.load_tif_folder(self.pic_folder)
        self.update_total_tracks_label()

    # --------------------- THEME FUNCTIONS ---------------------
    def set_theme(self, theme):
        """Set the UI theme (dark or light)."""
        if theme == "dark":
            bg = "#2E2E2E"
            fg = "white"
            btn_bg = "#007BFF"
            btn_fg = "white"
        else:  # light mode
            bg = "white"
            fg = "black"
            btn_bg = "#0056b3"
            btn_fg = "white"

        self.root.configure(bg=bg)
        self.style.theme_use("clam")
        self.style.configure("TButton", font=self.default_font, padding=6, background=btn_bg, foreground=btn_fg)
        self.style.map("TButton", background=[("active", "#3399FF")])
        self.style.configure("TLabel", font=self.default_font, background=bg, foreground=fg)
        self.style.configure("TCheckbutton", font=self.default_font, background=bg, foreground=fg)
        self.style.configure("TFrame", background=bg)
        self.style.configure("TEntry", fieldbackground="#F0F0F0", font=self.default_font)
        self.style.configure("TSpinbox", arrowsize=14, font=self.default_font)

        # Update canvas color
        if hasattr(self, 'canvas'):
            self.canvas.configure(bg="black" if theme == "dark" else "white")

    def toggle_theme(self):
        """Toggle between dark and light themes."""
        self.current_theme = "light" if self.current_theme == "dark" else "dark"
        self.mode_var.set("Dark" if self.current_theme == "dark" else "Light")
        self.set_theme(self.current_theme)

    def show_settings_window(self):
        """Display a window for changing key bindings."""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings - Key Bindings")
        settings_window.geometry("300x350")
        settings_window.configure(bg="#2E2E2E")

        tk.Label(settings_window, text="Change Key Bindings", font=("Helvetica", 14, "bold"),
                 fg="white", bg="#2E2E2E").pack(pady=10)

        # Store keybinds in a dict for easy modification
        keybinds = {
            "Start New Track": ["n", lambda e: self.start_new_track()],
            "Save CSV": ["s", lambda e: self.save_csv()],
            "Toggle Threshold": ["t", lambda e: self.toggle_threshold()],
            "Toggle Show Tracks": ["h", lambda e: self.toggle_show_saved_tracks()],
            "Toggle Detections": ["d", lambda e: self.toggle_detections()],
            "Auto Track Frame": ["a", lambda e: self.auto_track_current_frame()],
            "Auto Track All": ["A", lambda e: self.auto_track_until_lost()],
            "Select & Track": ["c", lambda e: self.select_and_track_particle()],
            "Next Frame": ["period", lambda e: self.goto_next_frame()],
            "Previous Frame": ["comma", lambda e: self.goto_prev_frame()],
        }

        frame = tk.Frame(settings_window, bg="#2E2E2E")
        frame.pack(fill="both", expand=True, pady=10)

        # Display each keybind with a change button
        for action, (key, callback) in keybinds.items():
            row = tk.Frame(frame, bg="#2E2E2E")
            row.pack(fill="x", pady=5)

            tk.Label(row, text=f"{action}: ", font=("Helvetica", 11), fg="white", bg="#2E2E2E").pack(side="left",
                                                                                                     padx=5)
            key_label = tk.Label(row, text=key, font=("Helvetica", 11, "bold"), fg="orange", bg="#2E2E2E")
            key_label.pack(side="left", padx=5)

            def make_change_fn(action_name=action, key_label=key_label, callback=callback):
                def change_key():
                    key_label.config(text="Press a key...")

                    def on_key_press(event):
                        new_key = event.keysym
                        key_label.config(text=new_key)
                        self.root.bind(new_key, callback)
                        settings_window.unbind("<Key>")

                    settings_window.bind("<Key>", on_key_press)

                return change_key

            ttk.Button(row, text="Change", command=make_change_fn()).pack(side="right", padx=5)

    def show_detector_settings(self):
        """Display a window for adjusting particle detector parameters."""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Particle Detector Settings")
        settings_window.geometry("400x400")
        settings_window.configure(bg="#2E2E2E")

        tk.Label(settings_window, text="Detector Parameters", font=("Helvetica", 14, "bold"),
                 fg="white", bg="#2E2E2E").pack(pady=10)

        frame = tk.Frame(settings_window, bg="#2E2E2E")
        frame.pack(fill="both", expand=True, pady=10, padx=20)

        # Min Area
        row1 = tk.Frame(frame, bg="#2E2E2E")
        row1.pack(fill="x", pady=5)
        tk.Label(row1, text="Min Area (pixels):", font=("Helvetica", 11), fg="white", bg="#2E2E2E").pack(side="left")
        min_area_var = tk.IntVar(value=self.particle_detector.min_area)
        tk.Spinbox(row1, from_=1, to=10000, textvariable=min_area_var, width=10,
                   font=self.default_font).pack(side="right")

        # Max Area
        row2 = tk.Frame(frame, bg="#2E2E2E")
        row2.pack(fill="x", pady=5)
        tk.Label(row2, text="Max Area (pixels):", font=("Helvetica", 11), fg="white", bg="#2E2E2E").pack(side="left")
        max_area_var = tk.IntVar(value=self.particle_detector.max_area)
        tk.Spinbox(row2, from_=1, to=50000, textvariable=max_area_var, width=10,
                   font=self.default_font).pack(side="right")

        # Min Density
        row3 = tk.Frame(frame, bg="#2E2E2E")
        row3.pack(fill="x", pady=5)
        tk.Label(row3, text="Min Density (0-1):", font=("Helvetica", 11), fg="white", bg="#2E2E2E").pack(side="left")
        min_density_var = tk.DoubleVar(value=self.particle_detector.min_density)
        tk.Spinbox(row3, from_=0.0, to=1.0, increment=0.05, textvariable=min_density_var, width=10,
                   font=self.default_font).pack(side="right")

        # Threshold Value
        row4 = tk.Frame(frame, bg="#2E2E2E")
        row4.pack(fill="x", pady=5)
        tk.Label(row4, text="Threshold (0-255):", font=("Helvetica", 11), fg="white", bg="#2E2E2E").pack(side="left")
        threshold_var = tk.IntVar(value=self.particle_detector.threshold_value)
        tk.Spinbox(row4, from_=0, to=255, textvariable=threshold_var, width=10,
                   font=self.default_font).pack(side="right")

        # Morphology
        row5 = tk.Frame(frame, bg="#2E2E2E")
        row5.pack(fill="x", pady=5)
        use_morphology_var = tk.BooleanVar(value=self.particle_detector.use_morphology)
        tk.Checkbutton(row5, text="Use Morphological Operations", variable=use_morphology_var,
                      font=("Helvetica", 11), fg="white", bg="#2E2E2E", selectcolor="#007BFF").pack(side="left")

        def apply_settings():
            self.particle_detector.min_area = min_area_var.get()
            self.particle_detector.max_area = max_area_var.get()
            self.particle_detector.min_density = min_density_var.get()
            self.particle_detector.threshold_value = threshold_var.get()
            self.particle_detector.use_morphology = use_morphology_var.get()
            self.show_frame()
            messagebox.showinfo("Settings Applied", "Detector settings have been updated!")
            settings_window.destroy()

        ttk.Button(frame, text="Apply Settings", command=apply_settings).pack(pady=20)

    # --------------------- TRACKING & UI FUNCTIONS ---------------------
    def validate_number(self, value_if_allowed):
        """Validate numeric input for frame jumping."""
        if value_if_allowed == "":
            return True
        return value_if_allowed.isdigit()

    def update_jump_entry(self):
        """Update the jump-to-frame entry box."""
        self.jump_var.set(str(self.current_index + 1))

    def toggle_threshold(self):
        """Enable or disable thresholding for image display."""
        self.threshold_enabled = not self.threshold_enabled
        self.toggle_thresh_button.config(text="Threshold (t)")
        self.show_frame()

    def toggle_detections(self):
        """Toggle display of detected particle centers."""
        self.show_frame()

    def estimate_next_position(self):
        """Estimate next particle position based on velocity and trajectory."""
        if len(self.tracking_history) < 2:
            return None
        
        # Use last few positions to estimate velocity
        recent_positions = self.tracking_history[-min(5, len(self.tracking_history)):]
        
        if len(recent_positions) < 2:
            return None
        
        # Calculate average velocity
        velocities_x = []
        velocities_y = []
        
        for i in range(1, len(recent_positions)):
            dx = recent_positions[i][0] - recent_positions[i-1][0]
            dy = recent_positions[i][1] - recent_positions[i-1][1]
            velocities_x.append(dx)
            velocities_y.append(dy)
        
        avg_vx = np.mean(velocities_x)
        avg_vy = np.mean(velocities_y)
        
        # Predict next position
        last_x, last_y = recent_positions[-1]
        predicted_x = last_x + avg_vx
        predicted_y = last_y + avg_vy
        
        return (predicted_x, predicted_y)

    def calculate_particle_score(self, particle, last_x, last_y, predicted_pos=None):
        """Calculate a score for how likely this particle is the correct one."""
        # Distance from last position
        dist_to_last = np.sqrt((particle.centroid_x - last_x)**2 + (particle.centroid_y - last_y)**2)
        
        # Confidence score from detector
        confidence = particle.confidence
        
        # Distance score (closer is better, normalized)
        max_reasonable_distance = 150
        distance_score = max(0, 1 - (dist_to_last / max_reasonable_distance))
        
        # If we have a predicted position, use it
        prediction_score = 0
        if predicted_pos is not None:
            pred_x, pred_y = predicted_pos
            dist_to_prediction = np.sqrt((particle.centroid_x - pred_x)**2 + (particle.centroid_y - pred_y)**2)
            prediction_score = max(0, 1 - (dist_to_prediction / max_reasonable_distance))
        
        # Combined score (weighted)
        if predicted_pos is not None:
            total_score = (0.4 * distance_score + 0.3 * confidence + 0.3 * prediction_score)
        else:
            total_score = (0.6 * distance_score + 0.4 * confidence)
        
        return total_score

    def find_best_matching_particle(self, particles, last_x, last_y, search_radius=None):
        """Find the best matching particle using multiple criteria."""
        if not particles:
            return None, 0
        
        # Get predicted position if we have enough history
        predicted_pos = self.estimate_next_position()
        
        # Calculate scores for all particles
        particle_scores = []
        for particle in particles:
            # If search radius specified, filter by distance
            if search_radius is not None:
                dist = np.sqrt((particle.centroid_x - last_x)**2 + (particle.centroid_y - last_y)**2)
                if dist > search_radius:
                    continue
            
            score = self.calculate_particle_score(particle, last_x, last_y, predicted_pos)
            particle_scores.append((particle, score))
        
        if not particle_scores:
            return None, 0
        
        # Sort by score and return best
        particle_scores.sort(key=lambda x: x[1], reverse=True)
        best_particle, best_score = particle_scores[0]
        
        return best_particle, best_score

    def interpolate_missing_frames(self, start_frame, end_frame, start_pos, end_pos):
        """Interpolate positions for missing frames."""
        interpolated = []
        frame_gap = end_frame - start_frame
        
        if frame_gap <= 1:
            return interpolated
        
        for i in range(1, frame_gap):
            t = i / frame_gap
            inter_x = round(start_pos[0] + t * (end_pos[0] - start_pos[0]))
            inter_y = round(start_pos[1] + t * (end_pos[1] - start_pos[1]))
            interp_frame = start_frame + i
            
            if 0 <= interp_frame - 1 < len(self.tif_paths):
                interp_file = os.path.basename(self.tif_paths[interp_frame - 1])
                interpolated.append({
                    "track": "track",
                    "frame": interp_frame,
                    "x": inter_x,
                    "y": inter_y,
                    "file": interp_file
                })
        
        return interpolated

    def is_particle_already_tracked(self, x, y, frame):
        """Check if a particle at this position has already been tracked in saved CSVs."""
        if not os.path.exists(self.track_folder):
            return False
        
        tolerance = 20  # pixels - particles within this distance are considered the same
        
        for file in os.listdir(self.track_folder):
            if file.endswith(".csv") and file.startswith("track_"):
                try:
                    parts = file[:-4].split("_")
                    start_frame = int(parts[1])
                    end_frame = int(parts[2])
                    
                    # Check if this frame is in the range of this saved track
                    if start_frame <= frame <= end_frame:
                        df = pd.read_csv(os.path.join(self.track_folder, file))
                        
                        # Check if any point in this track is near our position at this frame
                        for _, row in df.iterrows():
                            if int(row['frame']) == frame:
                                saved_x = int(row['x'])
                                saved_y = int(row['y'])
                                distance = np.sqrt((saved_x - x)**2 + (saved_y - y)**2)
                                
                                if distance < tolerance:
                                    return True
                except:
                    continue
        
        return False

    def select_and_track_particle(self):
        """Enable particle selection mode - user clicks on a particle to track it."""
        if not self.tif_paths or not (0 <= self.current_index < len(self.tif_paths)):
            messagebox.showwarning("No Frame", "No frame available.")
            return
        
        # Enable detection view if not already on
        if not self.show_detections.get():
            self.show_detections.set(True)
            self.show_frame()
        
        self.selecting_particle = True
        self.canvas.config(cursor="crosshair")
        messagebox.showinfo(
            "Select Particle",
            "Click on a particle to start tracking it.\n\n"
            "The system will automatically track that particle through all frames."
        )

    def handle_particle_selection_click(self, event):
        """Handle click when in particle selection mode."""
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        scale = getattr(self, 'display_scale', 1.0)
        click_x = int(canvas_x / scale)
        click_y = int(canvas_y / scale)
        
        # Load current frame and detect particles
        path = self.tif_paths[self.current_index]
        img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img_array is None:
            self.selecting_particle = False
            self.canvas.config(cursor="cross")
            messagebox.showwarning("Error", "Could not load image.")
            return
        
        particles = self.particle_detector.detect_particles_from_array(img_array)
        current_frame = self.current_index + 1
        
        # Filter untracked particles
        untracked_particles = []
        for particle in particles:
            if not self.is_particle_already_tracked(int(particle.centroid_x), int(particle.centroid_y), current_frame):
                untracked_particles.append(particle)
        
        if not untracked_particles:
            self.selecting_particle = False
            self.canvas.config(cursor="cross")
            messagebox.showwarning("No Particles", "No untracked particles detected on this frame.")
            return
        
        # Find the particle closest to the click
        min_dist = float('inf')
        selected_particle = None
        
        for particle in untracked_particles:
            dist = np.sqrt((particle.centroid_x - click_x)**2 + (particle.centroid_y - click_y)**2)
            if dist < min_dist:
                min_dist = dist
                selected_particle = particle
        
        # Check if click was reasonably close to a particle
        if min_dist > 50:  # 50 pixel tolerance
            messagebox.showwarning(
                "No Particle Selected",
                f"Click was {int(min_dist)} pixels from nearest particle.\n"
                "Please click closer to a particle."
            )
            return
        
        # Exit selection mode
        self.selecting_particle = False
        self.canvas.config(cursor="cross")
        
        # Start the track with this particle
        orig_x = int(selected_particle.centroid_x)
        orig_y = int(selected_particle.centroid_y)
        file_name = os.path.basename(path)
        
        self.tracks.append(
            {"track": "track", "frame": self.current_index + 1, "x": orig_x, "y": orig_y, "file": file_name})
        self.track_len_label.config(text=f"Track Length: {len(self.tracks)}")
        
        # Update tracking history
        self.tracking_history.append((orig_x, orig_y))
        if len(self.tracking_history) > self.max_tracking_history:
            self.tracking_history.pop(0)
        
        # Show confirmation
        response = messagebox.askyesno(
            "Start Tracking",
            f"Selected particle at ({orig_x}, {orig_y}).\n\n"
            "Track this particle through all frames automatically?"
        )
        
        if response:
            self.show_frame()
            self.root.update()
            # Start auto tracking from this particle
            self.auto_track_until_lost()
        else:
            self.show_frame()

    def auto_track_current_frame(self):
        """Automatically track the nearest detected particle on the current frame."""
        if not self.tif_paths or not (0 <= self.current_index < len(self.tif_paths)):
            messagebox.showwarning("No Frame", "No frame to track.")
            return
        
        path = self.tif_paths[self.current_index]
        
        # Load original image for detection
        img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img_array is None:
            messagebox.showwarning("Error", "Could not load image.")
            return
        
        # Detect particles
        particles = self.particle_detector.detect_particles_from_array(img_array)
        
        # Filter out particles that have already been tracked
        current_frame = self.current_index + 1
        untracked_particles = []
        for particle in particles:
            if not self.is_particle_already_tracked(int(particle.centroid_x), int(particle.centroid_y), current_frame):
                untracked_particles.append(particle)
        
        if not untracked_particles:
            # No untracked particles - either all tracked or none detected
            if self.tracks:
                if particles:
                    # There were particles but all already tracked
                    response = messagebox.askyesno(
                        "All Particles Tracked",
                        "All detected particles on this frame have already been tracked.\n\n"
                        "Would you like to save the current track?"
                    )
                else:
                    # No particles detected at all
                    response = messagebox.askyesno(
                        "Particle Lost",
                        "No particles detected on this frame. The particle may have left the screen.\n\n"
                        "Would you like to save the current track?"
                    )
                if response:
                    self.save_csv()
                else:
                    self.start_new_track()
            else:
                if particles:
                    messagebox.showinfo("All Tracked", "All particles on this frame have already been tracked.")
                else:
                    messagebox.showwarning("No Particles", "No particles detected on this frame.")
            return
        
        # Use untracked particles instead of all particles
        particles = untracked_particles
        
        # If we have a previous track point, find the nearest particle
        if self.tracks:
            last_x = self.tracks[-1]['x']
            last_y = self.tracks[-1]['y']
            
            # Find nearest particle to last position
            min_dist = float('inf')
            nearest_particle = None
            
            for particle in particles:
                dist = np.sqrt((particle.centroid_x - last_x)**2 + (particle.centroid_y - last_y)**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_particle = particle
            
            # Check if the nearest particle is too far away (likely a different particle)
            max_distance_threshold = 100  # pixels - adjust this value as needed
            if min_dist > max_distance_threshold:
                response = messagebox.askyesno(
                    "Particle Lost",
                    f"Nearest particle is {int(min_dist)} pixels away from last position.\n"
                    "The tracked particle may have left the screen.\n\n"
                    "Would you like to save the current track?"
                )
                if response:
                    self.save_csv()
                else:
                    self.start_new_track()
                return
            
            if nearest_particle:
                selected_particle = nearest_particle
            else:
                selected_particle = particles[0]
        else:
            # No previous track, select highest confidence particle
            selected_particle = particles[0]
        
        # Check if particle is near the edge of the image (about to leave)
        img_height, img_width = img_array.shape
        edge_margin = 50  # pixels from edge
        
        x_pos = selected_particle.centroid_x
        y_pos = selected_particle.centroid_y
        
        near_edge = (x_pos < edge_margin or x_pos > img_width - edge_margin or
                     y_pos < edge_margin or y_pos > img_height - edge_margin)
        
        # Add the selected particle to the track
        orig_x = int(selected_particle.centroid_x)
        orig_y = int(selected_particle.centroid_y)
        file_name = os.path.basename(path)
        
        # Handle interpolation if there's a gap
        last_frame = self.tracks[-1]['frame'] if self.tracks else None
        last_x = self.tracks[-1]['x'] if self.tracks else None
        last_y = self.tracks[-1]['y'] if self.tracks else None
        
        if last_frame and (self.current_index + 1 - last_frame > 1):
            frame_gap = self.current_index + 1 - last_frame
            for i in range(1, frame_gap):
                t = i / frame_gap
                inter_x = round(last_x + t * (orig_x - last_x))
                inter_y = round(last_y + t * (orig_y - last_y))
                interp_frame = last_frame + i
                interp_file = os.path.basename(self.tif_paths[interp_frame - 1])
                self.tracks.append(
                    {"track": "track", "frame": interp_frame, "x": inter_x, "y": inter_y, "file": interp_file})
        
        # Add the detected particle
        self.tracks.append(
            {"track": "track", "frame": self.current_index + 1, "x": orig_x, "y": orig_y, "file": file_name})
        self.track_len_label.config(text=f"Track Length: {len(self.tracks)}")
        
        # Check if particle is near edge after adding it
        if near_edge and len(self.tracks) > 3:  # Only warn if track has some length
            response = messagebox.askyesno(
                "Particle Near Edge",
                f"Particle is near the edge of the frame (x={orig_x}, y={orig_y}).\n"
                "It may leave the screen soon.\n\n"
                "Would you like to save the current track?"
            )
            if response:
                self.save_csv()
                return
        
        # Move to next frame
        gap = self.gap_var.get()
        if self.current_index + gap < len(self.tif_paths):
            self.current_index += gap
            self.root.after(50, self.show_frame)
        else:
            # Reached end of frames
            if self.tracks:
                response = messagebox.askyesno(
                    "End of Frames",
                    "Reached the end of the image sequence.\n\n"
                    "Would you like to save the current track?"
                )
                if response:
                    self.save_csv()

    def auto_track_until_lost(self):
        """Automatically track through multiple frames until particle is lost."""
        if not self.tif_paths or not (0 <= self.current_index < len(self.tif_paths)):
            messagebox.showwarning("No Frame", "No frame to track.")
            return
        
        # If no current track, start by getting first particle
        if not self.tracks:
            path = self.tif_paths[self.current_index]
            img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img_array is None:
                messagebox.showwarning("Error", "Could not load image.")
                return
            
            particles = self.particle_detector.detect_particles_from_array(img_array)
            current_frame = self.current_index + 1
            
            # Filter untracked particles
            untracked_particles = []
            for particle in particles:
                if not self.is_particle_already_tracked(int(particle.centroid_x), int(particle.centroid_y), current_frame):
                    untracked_particles.append(particle)
            
            if not untracked_particles:
                messagebox.showinfo("No Particles", "No untracked particles found on this frame.")
                return
            
            # Select highest confidence untracked particle
            selected_particle = untracked_particles[0]
            orig_x = int(selected_particle.centroid_x)
            orig_y = int(selected_particle.centroid_y)
            file_name = os.path.basename(path)
            
            self.tracks.append(
                {"track": "track", "frame": self.current_index + 1, "x": orig_x, "y": orig_y, "file": file_name})
            self.track_len_label.config(text=f"Track Length: {len(self.tracks)}")
            
            # Initialize tracking history
            self.tracking_history.append((orig_x, orig_y))
        
        # Enhanced tracking variables
        max_iterations = 10000
        iteration = 0
        consecutive_low_confidence = 0  # Track consecutive frames with low confidence
        frames_without_detection = 0  # Track frames where we couldn't find particle
        max_frames_without_detection = 5  # Allow missing particle for up to 5 frames
        adaptive_search_radius = 100  # Start with normal search radius
        
        while iteration < max_iterations:
            iteration += 1
            
            # Move to next frame
            gap = self.gap_var.get()
            if self.current_index + gap >= len(self.tif_paths):
                # Reached end of frames
                if self.tracks:
                    response = messagebox.askyesno(
                        "End of Frames",
                        f"Reached the end of the image sequence.\nTracked {len(self.tracks)} frames.\n\n"
                        "Would you like to save the current track?"
                    )
                    if response:
                        self.save_csv()
                break
            
            self.current_index += gap
            path = self.tif_paths[self.current_index]
            
            # Load and detect
            img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img_array is None:
                break
            
            particles = self.particle_detector.detect_particles_from_array(img_array)
            current_frame = self.current_index + 1
            
            # Filter untracked particles
            untracked_particles = []
            for particle in particles:
                if not self.is_particle_already_tracked(int(particle.centroid_x), int(particle.centroid_y), current_frame):
                    untracked_particles.append(particle)
            
            # Get last position
            last_x = self.tracks[-1]['x']
            last_y = self.tracks[-1]['y']
            last_frame = self.tracks[-1]['frame']
            
            # Use enhanced particle matching
            best_particle, confidence_score = self.find_best_matching_particle(
                untracked_particles, last_x, last_y, search_radius=adaptive_search_radius
            )
            
            if best_particle is None:
                frames_without_detection += 1
                
                # If we've been missing the particle for a while, try expanding search
                if frames_without_detection <= max_frames_without_detection:
                    # Increase search radius
                    adaptive_search_radius = min(200, adaptive_search_radius + 30)
                    
                    # Try again with expanded search
                    best_particle, confidence_score = self.find_best_matching_particle(
                        untracked_particles, last_x, last_y, search_radius=adaptive_search_radius
                    )
                    
                    if best_particle is None:
                        # Still no particle found, continue to next frame
                        # We'll interpolate later if we find it again
                        continue
                else:
                    # Lost particle for too many frames
                    response = messagebox.askyesno(
                        "Tracking Lost",
                        f"Lost track of particle for {frames_without_detection} consecutive frames.\n"
                        f"Tracked {len(self.tracks)} frames total.\n\n"
                        "Would you like to save the current track?"
                    )
                    if response:
                        self.save_csv()
                    break
            
            # Found a particle!
            frames_without_detection = 0
            adaptive_search_radius = 100  # Reset search radius
            
            # Check confidence score
            if confidence_score < 0.3:  # Low confidence threshold
                consecutive_low_confidence += 1
            else:
                consecutive_low_confidence = 0
            
            # If too many consecutive low confidence matches, warn user
            if consecutive_low_confidence > 10:
                response = messagebox.askyesno(
                    "Low Confidence Warning",
                    f"Tracking confidence has been low for {consecutive_low_confidence} frames.\n"
                    f"The particle may be lost or occluded.\n\n"
                    f"Tracked {len(self.tracks)} frames total.\n\n"
                    "Continue tracking?"
                )
                if not response:
                    # Ask to save
                    if messagebox.askyesno("Save Track", "Would you like to save the current track?"):
                        self.save_csv()
                    break
                else:
                    consecutive_low_confidence = 0  # Reset if user wants to continue
            
            selected_particle = best_particle
            
            # Check if near edge
            img_height, img_width = img_array.shape
            edge_margin = 50
            x_pos = selected_particle.centroid_x
            y_pos = selected_particle.centroid_y
            
            near_edge = (x_pos < edge_margin or x_pos > img_width - edge_margin or
                        y_pos < edge_margin or y_pos > img_height - edge_margin)
            
            # Add particle to track
            orig_x = int(selected_particle.centroid_x)
            orig_y = int(selected_particle.centroid_y)
            file_name = os.path.basename(path)
            
            # Handle interpolation for any skipped frames
            if current_frame - last_frame > 1:
                interpolated_points = self.interpolate_missing_frames(
                    last_frame, current_frame, (last_x, last_y), (orig_x, orig_y)
                )
                self.tracks.extend(interpolated_points)
                
                # Add interpolated positions to tracking history
                for point in interpolated_points:
                    self.tracking_history.append((point['x'], point['y']))
                    if len(self.tracking_history) > self.max_tracking_history:
                        self.tracking_history.pop(0)
            
            # Add current detection
            self.tracks.append(
                {"track": "track", "frame": current_frame, "x": orig_x, "y": orig_y, "file": file_name})
            self.track_len_label.config(text=f"Track Length: {len(self.tracks)}")
            
            # Update tracking history
            self.tracking_history.append((orig_x, orig_y))
            if len(self.tracking_history) > self.max_tracking_history:
                self.tracking_history.pop(0)
            
            # Update display periodically
            if iteration % 5 == 0:
                self.show_frame()
                self.root.update()
            
            # Check if near edge and stop (only if confident)
            if near_edge and len(self.tracks) > 3 and confidence_score > 0.5:
                response = messagebox.askyesno(
                    "Particle Near Edge",
                    f"Particle near edge at (x={orig_x}, y={orig_y}).\nTracked {len(self.tracks)} frames.\n\n"
                    "Would you like to save the current track?"
                )
                if response:
                    self.save_csv()
                break
        
        # Final display update
        self.show_frame()

    def load_tif_folder(self, folder):
        """Load .tif images from the specified folder."""
        self.tif_paths = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        ])
        self.current_index = 0
        self.tracks.clear()
        self.track_len_label.config(text="Track Length: 0")
        self.slider.config(from_=1, to=len(self.tif_paths))
        self.show_frame()

    def show_frame(self):
        """Display the current frame and overlays on the canvas."""
        self.canvas.delete("all")
        if not self.tif_paths or not (0 <= self.current_index < len(self.tif_paths)):
            return
        path = self.tif_paths[self.current_index]
        img = Image.open(path).convert("L")
        if self.threshold_enabled:
            thresh_val = self.threshold_slider.get()
            img = img.point(lambda p: 255 if p > thresh_val else 0)
        canvas_width = self.canvas.winfo_width() or 1
        canvas_height = self.canvas.winfo_height() or 1
        img_width, img_height = img.size
        scale = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        img_resized = img.resize((new_width, new_height), Image.NEAREST)
        self.tk_image = ImageTk.PhotoImage(img_resized)
        self.canvas.config(scrollregion=(0, 0, new_width, new_height))
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        self.display_scale = scale
        self.img_width, self.img_height = img_width, img_height
        guide_x = int(1080 * scale)
        self.canvas.create_line(guide_x, 0, guide_x, new_height, fill="blue", dash=(4, 2), width=1)
        
        # Show detected particles if enabled
        if self.show_detections.get():
            self.show_detected_particles()
        
        # Show manual tracking points
        points = [(int(t['x'] * scale), int(t['y'] * scale)) for t in self.tracks if
                  t['frame'] - 1 <= self.current_index]
        for i, (x, y) in enumerate(points):
            outline = "red" if i == len(points) - 1 else "gray"
            self.canvas.create_oval(x - 10, y - 10, x + 10, y + 10, outline=outline, width=1)
            if i > 0:
                x0, y0 = points[i - 1]
                line_color = "red" if i == len(points) - 1 else "gray"
                self.canvas.create_line(x0, y0, x, y, fill=line_color, width=1)
        
        self.frame_label.config(text=f"Frame: {self.current_index + 1}/{len(self.tif_paths)}")
        self.slider.set(self.current_index + 1)
        self.update_jump_entry()
        if self.show_saved_flag:
            self.show_saved_tracks()

    def show_detected_particles(self):
        """Overlay detected particle centers on current frame."""
        if not self.tif_paths or not (0 <= self.current_index < len(self.tif_paths)):
            return
        
        path = self.tif_paths[self.current_index]
        
        # Load original image for detection
        img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img_array is None:
            return
        
        # Detect particles
        particles = self.particle_detector.detect_particles_from_array(img_array)
        
        # Update detection count label
        self.detection_label.config(text=f"Detected: {len(particles)}")
        
        # Draw detected particles
        scale = getattr(self, 'display_scale', 1.0)
        for particle in particles:
            cx = int(particle.centroid_x * scale)
            cy = int(particle.centroid_y * scale)
            
            # Color based on confidence
            if particle.confidence > 0.7:
                color = "green"
            elif particle.confidence > 0.5:
                color = "yellow"
            else:
                color = "orange"
            
            # Draw centroid as a cross
            size = 8
            self.canvas.create_line(cx - size, cy, cx + size, cy, fill=color, width=2)
            self.canvas.create_line(cx, cy - size, cx, cy + size, fill=color, width=2)
            
            # Draw small circle around centroid
            radius = 12
            self.canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius,
                                   outline=color, width=2)
            
            # Optional: draw bounding box
            x, y, w, h = particle.bounding_box
            sx, sy = int(x * scale), int(y * scale)
            sw, sh = int(w * scale), int(h * scale)
            self.canvas.create_rectangle(sx, sy, sx + sw, sy + sh,
                                        outline=color, width=1, dash=(2, 2))

    def on_click(self, event):
        """Handle mouse click events for adding track points."""
        # If in particle selection mode, handle differently
        if self.selecting_particle:
            self.handle_particle_selection_click(event)
            return
        
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        scale = getattr(self, 'display_scale', 1.0)
        orig_x = int(canvas_x / scale)
        orig_y = int(canvas_y / scale)
        current_path = self.tif_paths[self.current_index]
        file_name = os.path.basename(current_path)
        gap = self.gap_var.get()
        last_frame = self.tracks[-1]['frame'] if self.tracks else None
        last_x = self.tracks[-1]['x'] if self.tracks else None
        last_y = self.tracks[-1]['y'] if self.tracks else None
        if last_frame and (self.current_index + 1 - last_frame > 1):
            frame_gap = self.current_index + 1 - last_frame
            for i in range(1, frame_gap):
                t = i / frame_gap
                inter_x = round(last_x + t * (orig_x - last_x))
                inter_y = round(last_y + t * (orig_y - last_y))
                interp_frame = last_frame + i
                interp_file = os.path.basename(self.tif_paths[interp_frame - 1])
                self.tracks.append(
                    {"track": "track", "frame": interp_frame, "x": inter_x, "y": inter_y, "file": interp_file})
        self.tracks.append(
            {"track": "track", "frame": self.current_index + 1, "x": orig_x, "y": orig_y, "file": file_name})
        self.track_len_label.config(text=f"Track Length: {len(self.tracks)}")
        self.current_index += gap
        self.root.after(50, self.show_frame)

    def on_mouse_move(self, event):
        """Update coordinate label based on mouse position."""
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        scale = getattr(self, 'display_scale', 1.0)
        orig_x = int(canvas_x / scale)
        orig_y = int(canvas_y / scale)
        self.coord_label.config(text=f"X: {orig_x}, Y: {orig_y}")

    def on_slider_move(self, value):
        """Handle slider movement for frame navigation."""
        index = int(value) - 1
        if 0 <= index < len(self.tif_paths):
            self.current_index = index
            self.show_frame()

    def goto_next_frame(self):
        """Advance to the next frame."""
        if self.current_index < len(self.tif_paths) - 1:
            self.current_index += 1
            self.show_frame()

    def goto_prev_frame(self):
        """Go back to the previous frame."""
        if self.current_index > 0:
            self.current_index -= 1
            self.show_frame()

    def on_mouse_scroll(self, event):
        """Handle mouse wheel scrolling for frame navigation (Windows/Mac)."""
        if event.delta > 0:
            self.goto_prev_frame()
        else:
            self.goto_next_frame()

    def on_mouse_scroll_linux(self, event):
        """Handle mouse wheel scrolling for frame navigation (Linux)."""
        if event.num == 4:
            self.goto_prev_frame()
        elif event.num == 5:
            self.goto_next_frame()

    def save_csv(self):
        """Save the current track to a CSV file."""
        if not self.tracks:
            messagebox.showwarning("No Data", "No tracking data to save.")
            return
        df = pd.DataFrame(self.tracks)[["track", "frame", "x", "y", "file"]]
        first_frame = df["frame"].min()
        last_frame = df["frame"].max()
        base_name = f"track_{first_frame:04d}_{last_frame:04d}.csv"
        os.makedirs(self.track_folder, exist_ok=True)
        base_path = os.path.join(self.track_folder, base_name)
        count = 1
        candidate_path = base_path
        while os.path.exists(candidate_path):
            candidate_path = base_path.replace(".csv", f"_{count}.csv")
            count += 1
        df.to_csv(candidate_path, index=False)
        self.csv_path = candidate_path
        self.start_new_track()
        self.update_total_tracks_label()

    def toggle_show_saved_tracks(self):
        """Toggle the display of saved tracks overlay."""
        self.show_saved_flag = not self.show_saved_flag
        self.show_saved_button.config(text="Hide Saved Tracks (h)" if self.show_saved_flag else "Show Saved Tracks (h)")
        self.show_frame()

    def show_saved_tracks(self):
        """Overlay saved tracks from CSV files on the current frame."""
        if not self.tif_paths:
            return
        current_frame = self.current_index + 1
        if not os.path.exists(self.track_folder):
            return
        for file in os.listdir(self.track_folder):
            if file.endswith(".csv") and file.startswith("track_"):
                try:
                    parts = file[:-4].split("_")
                    start_id = int(parts[1])
                    end_id = int(parts[2])
                    if start_id <= current_frame <= end_id:
                        df = pd.read_csv(os.path.join(self.track_folder, file))
                        for _, row in df.iterrows():
                            if int(row['frame']) == current_frame or (
                                    self.show_saved_history.get() and int(row['frame']) < current_frame):
                                x, y = int(row['x']), int(row['y'])
                                scale = getattr(self, 'display_scale', 1.0)
                                sx, sy = int(x * scale), int(y * scale)
                                if int(row['frame']) == current_frame:
                                    self.canvas.create_oval(sx - MARKER_RADIUS, sy - MARKER_RADIUS,
                                                            sx + MARKER_RADIUS, sy + MARKER_RADIUS,
                                                            outline="orange", width=MARKER_WIDTH, dash=(3, 2))
                                elif self.show_saved_history.get() and int(row['frame']) < current_frame:
                                    self.canvas.create_oval(sx - HISTORY_RADIUS, sy - HISTORY_RADIUS,
                                                            sx + HISTORY_RADIUS, sy + HISTORY_RADIUS,
                                                            outline="orange", width=1, fill="orange")
                except:
                    continue

    def start_new_track(self):
        """Clear the current track and start a new one."""
        self.tracks.clear()
        self.show_frame()

    def update_total_tracks_label(self):
        """Update the label showing the total number of saved tracks."""
        if not os.path.exists(self.track_folder):
            total = 0
        else:
            total = len([f for f in os.listdir(self.track_folder) if f.endswith(".csv") and f.startswith("track_")])
        self.total_tracks_label.config(text=f"Total Tracks: {total+1}")

    def jump_to_frame(self):
        """Jump to a specific frame based on user input."""
        try:
            frame_num = int(self.jump_var.get())
            if 1 <= frame_num <= len(self.tif_paths):
                self.current_index = frame_num - 1
                self.show_frame()
            else:
                messagebox.showwarning("Invalid Frame",
                                       f"Enter a frame between 1 and {len(self.tif_paths)}")
        except ValueError:
            messagebox.showwarning("Invalid Input", "Please enter a valid number.")


# ============= MAIN ENTRY POINT =============

def ask_for_folder(title, message):
    """
    Prompt the user to select a folder using a modal dialog.

    Parameters:
    -----------
    title : str
        The title of the popup window.
    message : str
        The message to display in the popup.

    Returns:
    --------
    str
        The selected folder path, or an empty string if cancelled.
    """
    popup = tk.Toplevel()
    popup.title(title)
    popup.geometry("350x150")
    popup.configure(bg="#2E2E2E")

    label = tk.Label(popup, text=message, font=("Helvetica", 12), fg="white", bg="#2E2E2E")
    label.pack(pady=15)

    folder_var = tk.StringVar()

    def browse_folder():
        folder = filedialog.askdirectory(title=title)
        if folder:
            folder_var.set(folder)
            popup.destroy()

    browse_button = ttk.Button(popup, text="Browse", command=browse_folder)
    browse_button.pack(pady=10)

    popup.grab_set()
    popup.wait_window()

    return folder_var.get()


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    pic_folder = ask_for_folder(
        "Select Dataset Folder",
        "Please select the dataset folder containing your particle images."
    )
    if not pic_folder:
        messagebox.showerror("No Folder Selected", "You must select a dataset folder.")
        root.destroy()
        exit()

    track_folder = ask_for_folder(
        "Select Track Folder",
        "Please select the folder where tracked particle data will be saved."
    )
    if not track_folder:
        messagebox.showerror("No Folder Selected", "You must select a track folder.")
        root.destroy()
        exit()

    root.deiconify()
    app = ManualTifTracker(root, pic_folder, track_folder)
    root.mainloop() 