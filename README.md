

---

# HandyTrack – Manual image tracking tool

Manual image tracking tool with hotkey controls, mouse scroller navigation, and interpolation to fill trajectories.

---

📖 **Manual Tracker – User Guide**
This GUI allows you to manually track particles across .tif images, view and edit tracks, and export tracking data into CSV files.

---
**Authors** 

Sathvik Muddasani -       

Hello, my name is Sathvik Muddasani, and I am one of the authors of this User Interface. I was born in Oregon and have moved around all over the country including Chicago, Minnesota twice, New York, Texas twice, and California before landing back in Minnesota for the third time. I love the summers, but I gotta say, I miss the Texas winters. I also miss rooting for a football team that didn't give me a heart attack every 5 seconds. The Vikings sure know how to keep us on the edge of our seats. 


I am currently a senior in Math and Science Academy doing full time PSEO classes at the University of Minnesota. I plan on pursuing Mechanical engineering in college and a masters in Aerospace.


Some of my hobbies are playing football with my friends and videogames. I also love going on vacation. I enjoy traveling to the corners of the planet to see and experience amazing things. I hope to continue to travel the world for as long as I can.



Grant Zeng - 

Hello, my name is Grant Zeng, and I’m one of the authors of this User Interface. I was born and raised in Minnesota, where I’ve spent most of my life balancing academics, athletics, and music. I’ve always loved the energy of Minnesota summers, especially when I’m on the court playing volleyball, but I could definitely do without the freezing winters. 

I’m currently a senior at Math and Science Academy, taking full-time PSEO classes at the University of Minnesota and Century College. I plan to study engineering in college, with a strong interest in applying problem-solving and creativity to real-world technology and design challenges.

Some of my hobbies are volleyball, weightlifting, and video games. One thing on my bucketlist is to go to space. 


---

## 1. 🖥️ Starting the Program

Run the script:

```
python tool_manual_tracking_joint.py
```

Two folder selection popups will appear:

* **Dataset Folder** → choose the folder containing your `.tif` images.
* **Track Folder** → choose the folder where CSV files will be saved.

The main window will then open.

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

### **Second Control Row**

* **Show Track History** → Displays the past positions of saved tracks.
* **Show/Hide Saved Tracks (h)** → Toggles overlay of saved tracks on the canvas.
* **Frame** → Shows the current frame index.
* **X, Y** → Shows the current mouse position over the image.
* **Total Tracks** → Shows how many track files exist in your save folder.

### **Canvas Area**

Displays the `.tif` image.

Shows overlays of:

* Current track (gray/red).
* Saved tracks (orange, dashed circles).
* Crosshair mouse cursor helps precision.
* **Guide Line** → A vertical dashed line at `x = 1080` (scaled), useful for alignment.

### **Frame Slider (bottom)**

* **Jump to Frame slider** → Drag to quickly move through frames.
* **Go to Frame Box** → Enter a frame number and press **Go** to jump directly.

### **Threshold Slider**

* Adjusts the threshold level when thresholding is enabled.
* Range: **0 → 255** (default: **128**).
* **Bring threshold to 0** if you want maximum contrast when locating faint particles.

---

## 3. 🖱️ Mouse Controls

* **Left Click** → Place a tracking point.

  * If there’s a gap from the last frame, intermediate points are interpolated automatically.
* **Mouse Move** → Updates the X,Y coordinate display.
* **Mouse Wheel (scroll)** → Navigate frames (forward/backward).

  * On Linux, scroll events are handled with Button-4 / Button-5.
* **Canvas Scrollbars** (bottom and right) → Manually pan when the image is larger than the view.

**Workflow Tip:**

1. Press **t** (or the Threshold button on top).
2. Bring **threshold slider to 0** for maximum sensitivity.
3. Click the **center of the particle** to place a point.

---

## 4. ⌨️ Keyboard Shortcuts

(Default bindings – can be changed in Settings)

* **n** → Start new track
* **s** or **Enter** → Save current track to CSV
* **t** → Toggle threshold view
* **h** → Toggle saved tracks overlay
* **,** → Previous frame
* **.** → Next frame

---

## 5. ⚙️ Settings (Key Bindings)

* Click **Settings** button to open the Key Bindings window.
* Each action shows its current key.
* Click **Change** → press a new key → the action is rebound instantly.

Configurable actions include:

* New Track
* Save CSV
* Toggle Threshold
* Toggle Saved Tracks
* Next Frame
* Previous Frame

---

## 6. 💾 Saving & Managing Tracks

* **Save CSV** creates a new CSV file:

  * Format: `track_XXXX_YYYY.csv`

    * `XXXX` = first frame in track
    * `YYYY` = last frame in track
  * If a file with the same name exists, a number is appended (e.g., `_1`, `_2`).

* Saved tracks can be overlaid on images by toggling **Show Saved Tracks (h)**.

* **Total Tracks counter** in the GUI shows how many saved track files exist in your selected track folder.

---

## 7. 🔍 Visualization Features

* **Track History checkbox** → Draws all previous points of saved tracks up to the current frame.
* **Guide Line** → A vertical dashed line is drawn at `x = 1080` (scaled), useful for alignment.
* **Threshold Mode** → Helps highlight particles against the background.

  * Use the **Threshold Slider** to fine-tune (0–255).

---

## 8. 🚨 Error Handling

* If you try to **save with no data** → Warning message is shown.
* If you enter an **invalid frame number** in "Go to Frame" → Warning message is shown.

---

## ✅ Quick Workflow

1. Open dataset and track folders.
2. Move to the frame of interest (**.** or slider).
3. Press **t** to enable threshold view.
4. Bring threshold slider to **0** if needed.
5. Click on the **center of the particle** → point is added.
6. Advance frames (**comma / period**) and keep clicking.
7. When finished → **Save CSV (s)**.
8. Start a **New Track (n)** and repeat.

---

