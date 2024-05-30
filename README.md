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
```
Using as a Module in Your Project

You can also integrate PoseTrackingModule into your own project. Here is an example of how to use the PoseDetector class:

```python
from PoseTrackingModule import PoseDetector
import cv2

def main():
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.find_pose(img)
        lmList = detector.get_landmarks(img)
        if len(lmList) != 0:
            print(lmList[0])

        cv2.imshow('Image', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

PoseDetector Class

Initialization

```python detector = PoseDetector(mode=False, model_complexity=1, smooth_landmarks=True, enable_segmentation=False, smooth_segmentation=True, detectionCon=0.5, trackCon=0.5)```

Methods
`find_pose(img, draw=True)`: Processes the image to detect pose landmarks and optionally draws them.
`get_landmarks(img, draw=True)`: Returns a list of detected landmark positions and optionally draws circles on them.
Integration
The PoseTrackingModule can be easily integrated into larger projects such as fitness applications, gesture-controlled interfaces, and more. Its modular design allows you to customize and extend its capabilities based on your project's requirements.

License
This project is licensed under the MIT `License`. See the LICENSE file for more details.

Feel free to contribute to this project by submitting issues or pull requests.

Happy coding!
