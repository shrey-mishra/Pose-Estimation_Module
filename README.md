# Pose Estimation Module

PoseTrackingModule is a Python module for real-time pose detection using OpenCV and MediaPipe. This module captures video input, processes each frame to detect human poses, and displays the results with annotated landmarks. It is designed to be easy to integrate into any larger project that requires pose detection capabilities.

Features

- Real-time pose detection using MediaPipe.
- Customizable detection and tracking confidence levels.
- Annotated display of detected pose landmarks.
- Frame rate (FPS) calculation and display.

Installation

Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

Requirements

- OpenCV
- MediaPipe

Step-by-Step Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/PoseTrackingModule.git
    cd PoseTrackingModule
    ```

2. Create and activate a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install required packages:

    ```bash
    pip install -r requirements.txt
    ```

USAGE

### Running the Module

To run the module and start pose detection from your webcam, execute the following command:

```bash
python PoseTrackingModule.py
