# Multi-Modal Interaction System

This project is a **Multi-Modal Interaction System** that enables hands-free control of a computer's mouse using **eye tracking**, **head movements**, and **hand gestures**. The system detects facial landmarks for controlling the mouse with eye blinks and head movement and uses hand landmarks for gesture-based mouse control.

## Output
![Screenshot 2024-10-06 111157](https://github.com/user-attachments/assets/b9ef5daa-d469-405a-9fdf-cb194ad7736d)
![Screenshot 2024-10-06 111243](https://github.com/user-attachments/assets/e47fcc10-35e0-4371-b46c-006a8cedb8ee)



## Features
- **Eye Tracking and Blink Detection**: Move the cursor based on head movements and perform left and right clicks by blinking.
- **Hand Gesture Recognition**: Move the cursor with your hand, and use finger gestures to simulate left and right mouse clicks.
- **Smooth Cursor Movements**: Incorporates sensitivity adjustments and smoothing to enhance user experience.
- **Hybrid Control**: Switches seamlessly between eye control (when no hands are detected) and hand control (when a hand is detected).

## Technologies Used
- **Python**
- **OpenCV** (for video capture and image processing)
- **Mediapipe** (for facial and hand landmark detection)
- **PyAutoGUI** (for controlling the mouse)
- **Pynput** (for simulating mouse clicks)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/toshit111/Multi-Model-Interaction-system.git
    cd Multi-Model-Interaction-system
    ```

2. Install dependencies:

    ```bash
    pip install opencv-python mediapipe pyautogui pynput
    ```

3. Run the project:

    ```bash
    python multi_modal_interaction_system.py
    ```

## How It Works
- **Eye Control**: Detects facial landmarks using Mediapipe's Face Mesh model and tracks head movements to move the cursor. Blinks are detected for left and right clicks.
- **Hand Control**: Detects hand landmarks using Mediapipe’s Hand model. The index finger controls cursor movement, while thumb and finger gestures trigger left and right mouse clicks.

## Usage Instructions
- Ensure your webcam is properly connected.
- To switch between eye and hand control:
  - **Hand Control**: When a hand is detected, the system will follow hand gestures.
  - **Eye Control**: When no hand is detected, the system reverts to eye-based control.

- **Left Click**: Blink your left eye for a left-click, or pinch the thumb and index finger for a click.
- **Right Click**: Blink your right eye for a right-click, or pinch with your middle finger for a right-click.

## Contact
toshitfirake2003@gmail.com
