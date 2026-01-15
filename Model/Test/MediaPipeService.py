import cv2
import mediapipe as mp
from LandmarkDrawer import LandmarkDrawer
from Utilities import Utilities


class MediaPipeService:
    """
    Centralizes MediaPipe Holistic initialization and detection logic.
    Provides a consistent way to process frames and draw landmarks.
    """

    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawer = LandmarkDrawer(self.mp_drawing, self.mp_holistic)
        self.logger = Utilities.setup_logging()

    def mediapipe_detection(self, frame, model):
        """Processes a single frame using the provided MediaPipe model."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = model.process(frame_rgb)
        frame_rgb.flags.writeable = True
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), results

    def draw_landmarks(self, frame, results):
        """Draws detected landmarks on the frame."""
        self.drawer.draw(frame, results)
