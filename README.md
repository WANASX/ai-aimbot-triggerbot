# GamerFun AI Menu

**GamerFun AI Menu** is an advanced tool that combines AI-powered aimbot, triggerbot, and recoil control—optimized for Rainbow Six Siege and many other top shooter games. By integrating state-of-the-art computer vision with Logitech driver support, GamerFun AI Menu provides precision targeting, smooth mouse movements, and customizable recoil profiles while mimicking human behavior to help avoid detection.

---

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Performance & Requirements](#performance--requirements)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

- **AI Aimbot & Triggerbot**
  - Real-time enemy detection using a YOLO model.
  - Customizable settings including aim speed and target areas (head, neck, chest, legs).
  - Automatically fires when an enemy enters the crosshair (triggerbot).

- **Recoil Control**
  - Auto agent detection (triggered via F5) to automatically load the correct recoil profile.
  - Manual agent selection from the UI for precise recoil management.
  - Integrated with Logitech’s driver for safe and reliable recoil control.

- **User-Friendly Interface**
  - Easy-to-use UI built with PyQt5.
  - Real-time adjustments without editing configuration files.
  - Customizable activation keys:
    - **Left Alt:** Hold to activate Aimbot & Triggerbot.
    - **Caps Lock:** Toggle triggerbot active state.
    - **F5:** Trigger auto agent detection for recoil control.

- **Randomized Movements**
  - Incorporates slight random offsets in mouse movement and firing delays to mimic natural human behavior and reduce the risk of detection.

- **Optimized Performance**
  - Capable of running in the background with minimal impact on game performance.
  - Supports GPU acceleration with CUDA (recommended) for drastically reduced detection times.

---

## How It Works

1. **Detection:**  
   The tool continuously captures screen frames (using [mss](https://pypi.org/project/mss/)) and uses a YOLO (You Only Look Once) model (via [ultralytics](https://pypi.org/project/ultralytics/)) to detect enemy targets.

2. **Aiming:**  
   Once an enemy is detected, the program calculates the target coordinates (e.g., neck, head, etc.) and moves the mouse smoothly towards the target using Logitech driver integration.

3. **Shooting:**  
   If the enemy falls within the designated crosshair area and if triggerbot is enabled, the tool automatically simulates a mouse click to fire.

4. **Recoil Control:**  
   For weapons with significant recoil, the tool uses pre-defined recoil profiles—either auto-detected or manually selected—to counteract recoil during sustained fire.

---

## Installation

### Prerequisites

- **Operating System:** Windows (the tool uses Windows-specific libraries such as `win32api`)
- **Python 3.8+**
- **CUDA 12.4** (Highly recommended for GPU acceleration; otherwise, detection runs on CPU and is slower)
- **Tesseract OCR**
  - Download and install from the [official GitHub repository](https://github.com/tesseract-ocr/tesseract)
  - Ensure `Tesseract.exe` is added to your system PATH

### Steps

1. **Clone the Repository:**

       git clone https://github.com/yourusername/GamerFun-AI-Menu.git
       cd GamerFun-AI-Menu

2. **Install Python Dependencies:**

   It is recommended to use a virtual environment.

       python -m venv venv
       source venv/Scripts/activate  # On Windows use: venv\Scripts\activate
       pip install --upgrade pip
       pip install -r requirements.txt

3. **Setup Logitech G HUB:**

   - Download and install **Logitech G HUB** (version 2021-10-8013 recommended) even if you do not own a Logitech mouse.
   - **Block Logitech G HUB from Internet Access:**  
     Use your firewall settings to block internet access to Logitech G HUB to prevent automatic updates.

4. **Additional Setup:**

   - Ensure that the required model file (`model1.pt`) and configuration files (`config.json`, `profiles.json`, and the Logitech DLL `ghub_mouse.dll`) are placed in the `libs` directory.
   - Run the provided installer batch file if available (e.g., `Installer.bat`) to complete any extra setup steps.

---

## Usage

1. **Launch the Application:**

   Run the main executable (e.g., `GamerFun.exe`) or start the Python script:

       python main.py

2. **In-Game Activation:**

   - **Hold Left Alt:** Activates the Aimbot and Triggerbot functions.
   - **Press Caps Lock:** Toggle the triggerbot’s active state.
   - **Press F5:** Auto-detect your in-game agent for recoil control.

3. **Adjust Settings on the Fly:**

   Use the intuitive UI to adjust aim speed, select target areas, choose recoil profiles, and modify other settings without editing the configuration files manually.

---

## Configuration

The configuration file (`config.json`) located in the `libs` folder holds all adjustable parameters. This includes settings for:

- **Aimbot:** Enable/disable, aim speed, target choice (e.g., neck)
- **Triggerbot:** Activation key and trigger settings
- **Recoil Control:** Agent selection and recoil compensation values
- **Screen Regions:** Customizable coordinates for different aspect ratios

If the configuration file is not found, a default configuration will be automatically generated.

---

## Performance & Requirements

- **GPU Acceleration:**  
  For maximum scanning speed, install **CUDA 12.4**.
  - **With CUDA (GPU):** Approx. **5ms** detection times
  - **Without CUDA (CPU):** Approx. **30ms** detection times

- **System Impact:**  
  Optimized to run in the background with minimal performance impact on your game.

---

## Troubleshooting

- **Model Not Found:**  
  Ensure that `model1.pt` exists in the `libs` folder and that the configuration file points to the correct path.

- **Tesseract Issues:**  
  Confirm that Tesseract OCR is installed correctly and that its executable is included in your system PATH.

- **Driver Integration:**  
  If recoil control isn’t working, verify that Logitech G HUB is installed, the DLL (`ghub_mouse.dll`) is present in the `libs` folder, and that you have blocked G HUB from updating.

For further assistance, please check the issue tracker or open a new issue on GitHub.

---

## License

This project is provided "as is" without any warranty. Please review the [LICENSE](LICENSE) file for more details.

---

## Step-By-Step Video Guide

Watch our video guide for a detailed walkthrough on installation and usage:  
[Step-By-Step Video Guide](https://www.youtube.com/watch?v=RbwX7uay4_Q)  <!-- Replace with actual video link when available -->

---

Enjoy gaming with precision and enhanced control using GamerFun AI Menu!
