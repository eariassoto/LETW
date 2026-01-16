import cv2
import mediapipe as mp
from Utilities import Utilities


class HolisticProcessor:
    """
    Encapsulates the MediaPipe Holistic model lifecycle and processing.
    Used as a context manager.
    """

    def __init__(self, holistic_class, confidence):
        self.holistic_class = holistic_class
        self.confidence = confidence
        self.model = None

    def __enter__(self):
        self.model = self.holistic_class(
            min_detection_confidence=self.confidence, min_tracking_confidence=self.confidence
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.model:
            self.model.close()

    def process(self, frame):
        """Processes a single frame using the internal MediaPipe model."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.model.process(frame_rgb)
        frame_rgb.flags.writeable = True
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), results


class MediaPipeService:
    """
    Centralizes MediaPipe Holistic initialization and detection logic.
    Provides a consistent way to process frames and draw landmarks.
    """

    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.logger = Utilities.setup_logging()

    def start_holistic_session(self, confidence):
        """Creates and returns a HolisticProcessor context manager."""
        return HolisticProcessor(self.mp_holistic.Holistic, confidence)

    def draw_landmarks(self, frame, results):
        """Draws detected landmarks on the frame."""
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.face_landmarks,
                self.mp_holistic.FACEMESH_TESSELATION,
                self.mp_drawing.DrawingSpec(color=(103, 207, 245), thickness=1, circle_radius=1),
                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1),
            )

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)

        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)

        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
