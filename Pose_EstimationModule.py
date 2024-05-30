# Import necessary libraries
import cv2  # OpenCV for image and video processing
import mediapipe as mp  # MediaPipe for pose tracking
import time  # Time library for FPS calculation


# Define a class for pose detection
class PoseDetector:
    def __init__(self, mode=False, model_complexity=1, smooth_landmarks=True,
                 enable_segmentation=False, smooth_segmentation=True,
                 detectionCon=0.5, trackCon=0.5):
        """
        Initialize the PoseDetector with various configuration parameters.

        Parameters:
        mode (bool): Whether to treat the input images as a batch of static and possibly unrelated images.
        model_complexity (int): Complexity of the pose landmark model: 0, 1, or 2.
        smooth_landmarks (bool): Whether to filter landmarks across different input images to reduce jitter.
        enable_segmentation (bool): Whether to enable segmentation.
        smooth_segmentation (bool): Whether to smooth the segmentation mask.
        detectionCon (float): Minimum confidence value ([0.0, 1.0]) for the detection to be considered successful.
        trackCon (float): Minimum confidence value ([0.0, 1.0]) for the landmark-tracking model to be considered successful.
        """
        self.mode = mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Initialize MediaPipe's Pose module
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     model_complexity=self.model_complexity,
                                     smooth_landmarks=self.smooth_landmarks,
                                     enable_segmentation=self.enable_segmentation,
                                     smooth_segmentation=self.smooth_segmentation,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)

    def find_pose(self, img, draw=True):
        """
        Process an image to detect pose landmarks and optionally draw them.

        Parameters:
        img (ndarray): The input image.
        draw (bool): Whether to draw landmarks on the image.

        Returns:
        ndarray: The processed image with or without drawn landmarks.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def get_landmarks(self, img, draw=True):
        """
        Get the pose landmarks from the processed image.

        Parameters:
        img (ndarray): The input image.
        draw (bool): Whether to draw circles on the landmarks.

        Returns:
        list: A list of landmark positions [id, x, y].
        """
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return lmList


def main():
    """
    The main function to capture video from the webcam, process it to detect poses,
    and display the results with FPS.
    """
    pTime = 0  # Previous time for FPS calculation
    cap = cv2.VideoCapture(0)  # Open the default webcam
    detector = PoseDetector()  # Initialize the PoseDetector

    while True:
        success, img = cap.read()  # Read a frame from the webcam
        if not success:
            break

        img = detector.find_pose(img)  # Detect pose and optionally draw landmarks
        lmList = detector.get_landmarks(img)  # Get the list of landmark positions
        if len(lmList) != 0:
            print(lmList[0])  # Example of printing the first landmark's details

        # Calculate FPS
        cTime = time.time()  # Get the current time
        fps = 1 / (cTime - pTime)  # Calculate FPS as the inverse of frame time
        pTime = cTime  # Update the previous time

        # Display FPS on the image
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        # Show the processed image in a window
        cv2.imshow('Image', img)

        # Optional: Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
