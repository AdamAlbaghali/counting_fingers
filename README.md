# Counting Fingers

This Python project uses OpenCV to count the number of fingers shown in a live video feed from a webcam.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/AdamAlbaghali/counting_fingers.git
    ```

2. Install the required libraries:

    ```bash
    pip install opencv-python
    ```

## Usage

1. Run the `fingerdetection.py` script:

    ```bash
    python fingerdetection.py
    ```

2. Place your hand in front of the webcam.
3. The program will detect and count the number of fingers shown in the video feed.

## How it Works

The `count_digits` function analyzes the video frames to detect fingers using contour detection. It then counts the number of fingers detected and displays the count on the screen.

