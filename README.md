# handytrack
HandyTrack – Manual image tracking tool with hotkey controls, mouse scroller navigation, and interpolation to fill trajectories.

---

# 📖 Manual Tracker – User Guide

This GUI allows you to manually track particles across `.tif` images, view and edit tracks, and export tracking data into CSV files.

---

## 1. 🖥️ Starting the Program

1. Run the script (`python your_script.py`).
2. Two folder selection popups will appear:

   * **Dataset Folder** → choose the folder containing your `.tif` images.
   * **Track Folder** → choose the folder where CSV files will be saved.
3. The main window will then open.

---

## 2. 🗂️ Main Window Overview

The GUI has the following sections:

### **Top Control Row**

* **New Track (n)** → Starts a new track (clears the current path).
* **Save CSV (s)** → Saves the current track to the track folder.
* **Threshold (t)** → Toggles threshold view of the image (binary contrast).
* **Settings** → Opens a window to rebind keyboard shortcuts.
* **Skip Gap** → Choose the number of frames to skip after each point placement.
* **Track Length** → Displays how many points are in the current track.
* **Dark Mode toggle** → Switches between Dark and Light themes.

---

### **Second Control Row**

* **Show Track History** → Displays the past positions of saved tracks.
* **Show/Hide Saved Tracks (h)** → Toggles overlay of saved tracks on the canvas.
* **Frame** → Shows the current frame index.
* **X, Y** → Shows the current mouse position over the image.
* **Total Tracks** → Shows how many track files exist in your save folder.

---

### **Canvas Area**

* Displays the `.tif` image.
* Shows overlays of:

  * Current track (gray/red).
  * Saved tracks (orange, dashed circles).
* Crosshair mouse cursor helps precision.

---

### **Frame Slider**

* `Jump to Frame` slider → Drag to quickly move through frames.

---

### **Go to Frame Box**

* Enter a frame number and press **Go** to jump directly.

---

### **Threshold Slider**

* Adjusts the threshold level when thresholding is enabled.

---

## 3. 🖱️ Mouse Controls

* **Left Click** → Place a tracking point.

  * If there’s a gap from the last frame, intermediate points are interpolated automatically.
* **Mouse Move** → Updates the **X,Y coordinate display**.
* **Mouse Wheel (scroll)** → Navigate frames (forward/backward).

  * On Linux, scroll events are handled with Button-4 / Button-5.

---

## 4. ⌨️ Keyboard Shortcuts

(Default bindings – can be changed in **Settings**)

* `n` → Start new track
* `s` or `Enter` → Save current track to CSV
* `t` → Toggle threshold view
* `h` → Toggle saved tracks overlay
* `,` → Previous frame
* `.` → Next frame

---

## 5. ⚙️ Settings (Key Bindings)

* Click **Settings** button to open the Key Bindings window.
* Each action shows its current key.
* Click **Change** → press a new key → the action is rebound instantly.

---

## 6. 💾 Saving & Managing Tracks

* **Save CSV** creates a new CSV file:

  * Format: `track_XXXX_YYYY.csv`

    * `XXXX` = first frame in track
    * `YYYY` = last frame in track
  * If a file with the same name exists, a number is appended (e.g. `_1`, `_2`).
* Saved tracks can be overlaid on images by toggling **Show Saved Tracks (h)**.

---

## 7. 🔍 Visualization Features

* **Track History** checkbox → Draws all previous points of saved tracks up to the current frame.
* **Guide Line** → A vertical dashed line is drawn at x=1080 (scaled), useful for alignment.
* **Threshold Mode** → Helps highlight particles against the background.

---

## 8. 🚨 Error Handling

* If you try to save with no data → Warning message is shown.
* If you enter an invalid frame number in "Go to Frame" → Warning message is shown.

---

# ✅ Quick Workflow

1. Open dataset and track folders.
2. Move to the frame of interest (`.` or slider).
3. Click on the particle → point is added.
4. Advance frames (`,` / `.`) and keep clicking.
5. When finished → **Save CSV (s)**.
6. Start a **New Track (n)** and repeat.

---

Do you want me to also include **screenshots with callouts** (I can generate them) so this reads like a polished PDF-style manual?



Authors - 

Grant Zeng: Welcome to handyman! I am a senior doing 
PSEO classes at the UMN. Some hobbies of mine are playing volleyball, Valorant, and getting boba!  

Sathvik Muddasani: I am a Senior at the Math and Science Academy currently taking classes at the Univeristy of Minnesota. 
