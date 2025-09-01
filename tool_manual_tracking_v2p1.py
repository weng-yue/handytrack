import os
import pandas as pd
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox

# path of tracks and dataset
PIC_FOLDER = r"/Users/suman/Downloads/GitHub/intern_flowtrack/FFIL/xy_dataset_2"
TRACK_FOLDER = r"/Users/suman/Downloads/GitHub/intern_flowtrack/FFIL/tracks_dataset_2"

IMAGE_EXTENSIONS = [".tif"]

# Visualization parameters
MARKER_RADIUS = 5
MARKER_WIDTH = 2
HISTORY_RADIUS = 2
HISTORY_LINE_WIDTH = 2


class ManualTifTracker:
    def __init__(self, root):
        self.root = root
        self.root.title("Manual Tracker with Threshold, Path, and Scaling")
        self.root.geometry("800x800")

        # ---- THEME STYLING ----
        self.style = ttk.Style()
        self.default_font = ("Helvetica", 11)  # define BEFORE set_theme
        self.current_theme = "dark"
        self.set_theme(self.current_theme)

        # Init vars BEFORE loading images
        self.show_saved_flag = False
        self.show_saved_history = tk.BooleanVar(value=False)
        self.tif_paths = []
        self.current_index = 0
        self.tracks = []
        self.csv_path = None
        self.tk_image = None
        self.threshold_enabled = False

        # First control row
        control_row1 = ttk.Frame(self.root)
        control_row1.pack(fill="x", side="top", pady=3)
        ttk.Button(control_row1, text="New Track (n)", command=self.start_new_track).pack(side="left", padx=5)
        ttk.Button(control_row1, text="Save CSV (s)", command=self.save_csv).pack(side="left", padx=5)
        self.toggle_thresh_button = ttk.Button(control_row1, text="Threshold (t)", command=self.toggle_threshold)
        self.toggle_thresh_button.pack(side="left", padx=5)
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
        self.show_saved_button = ttk.Button(control_row2, text="Show Saved Tracks (h)", command=self.toggle_show_saved_tracks)
        self.show_saved_button.pack(side="left", padx=5)
        self.frame_label = ttk.Label(control_row2, text="Frame: 0")
        self.frame_label.pack(side="left", padx=10)
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
        self.threshold_slider.set(128)
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

        self.load_tif_folder(PIC_FOLDER)
        self.update_total_tracks_label()

    # --------------------- THEME FUNCTIONS ---------------------
    def set_theme(self, theme):
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
        self.current_theme = "light" if self.current_theme == "dark" else "dark"
        self.mode_var.set("Dark" if self.current_theme == "dark" else "Light")
        self.set_theme(self.current_theme)

    # --------------------- TRACKING & UI FUNCTIONS ---------------------
    def validate_number(self, value_if_allowed):
        if value_if_allowed == "":
            return True
        return value_if_allowed.isdigit()

    def update_jump_entry(self):
        self.jump_var.set(str(self.current_index + 1))

    def toggle_threshold(self):
        self.threshold_enabled = not self.threshold_enabled
        self.toggle_thresh_button.config(text="Threshold (t)")
        self.show_frame()

    def load_tif_folder(self, folder):
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

    def on_click(self, event):
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
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        scale = getattr(self, 'display_scale', 1.0)
        orig_x = int(canvas_x / scale)
        orig_y = int(canvas_y / scale)
        self.coord_label.config(text=f"X: {orig_x}, Y: {orig_y}")

    def on_slider_move(self, value):
        index = int(value) - 1
        if 0 <= index < len(self.tif_paths):
            self.current_index = index
            self.show_frame()

    def goto_next_frame(self):
        if self.current_index < len(self.tif_paths) - 1:
            self.current_index += 1
            self.show_frame()

    def goto_prev_frame(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_frame()

    def on_mouse_scroll(self, event):
        if event.delta > 0:
            self.goto_prev_frame()
        else:
            self.goto_next_frame()

    def on_mouse_scroll_linux(self, event):
        if event.num == 4:
            self.goto_prev_frame()
        elif event.num == 5:
            self.goto_next_frame()

    def save_csv(self):
        if not self.tracks:
            messagebox.showwarning("No Data", "No tracking data to save.")
            return
        df = pd.DataFrame(self.tracks)[["track", "frame", "x", "y", "file"]]
        first_frame = df["frame"].min()
        last_frame = df["frame"].max()
        base_name = f"track_{first_frame:04d}_{last_frame:04d}.csv"
        os.makedirs(TRACK_FOLDER, exist_ok=True)
        base_path = os.path.join(TRACK_FOLDER, base_name)
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
        self.show_saved_flag = not self.show_saved_flag
        self.show_saved_button.config(text="Hide Saved Tracks (h)" if self.show_saved_flag else "Show Saved Tracks (h)")
        self.show_frame()

    def show_saved_tracks(self):
        if not self.tif_paths:
            return
        current_frame = self.current_index + 1
        if not os.path.exists(TRACK_FOLDER):
            return
        for file in os.listdir(TRACK_FOLDER):
            if file.endswith(".csv") and file.startswith("track_"):
                try:
                    parts = file[:-4].split("_")
                    start_id = int(parts[1])
                    end_id = int(parts[2])
                    if start_id <= current_frame <= end_id:
                        df = pd.read_csv(os.path.join(TRACK_FOLDER, file))
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
        self.tracks.clear()
        self.show_frame()

    def update_total_tracks_label(self):
        if not os.path.exists(TRACK_FOLDER):
            total = 0
        else:
            total = len([f for f in os.listdir(TRACK_FOLDER) if f.endswith(".csv") and f.startswith("track_")])
        self.total_tracks_label.config(text=f"Total Tracks: {total+1}")

    def jump_to_frame(self):
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


if __name__ == "__main__":
    root = tk.Tk()
    app = ManualTifTracker(root)
    root.mainloop()
fghghfdffdgdfg