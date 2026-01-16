from __future__ import annotations

from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from Utilities import Utilities


class MediaPipeService:
    """
    Centralizes MediaPipe Holistic initialization and detection logic.
    Provides a consistent way to process frames and draw landmarks.
    """

    def __init__(self) -> None:
        self.logger = Utilities.setup_logging()

    def create_holistic(self, confidence: float) -> Any:
        """Creates and returns a MediaPipe Holistic model instance."""
        return mp.solutions.holistic.Holistic(min_detection_confidence=confidence, min_tracking_confidence=confidence)

    def process(self, model: Any, frame: np.ndarray, draw: bool = False) -> tuple[np.ndarray, Any]:
        """Processes a single frame using the provided MediaPipe model."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = model.process(frame_rgb)
        frame_rgb.flags.writeable = True
        image = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if draw:
            self.draw_landmarks(image, results)

        return image, results

    def draw_landmarks(self, frame: np.ndarray, results: Any) -> None:
        """Draws detected landmarks on the frame."""
        mp_drawing = mp.solutions.drawing_utils
        mp_holistic = mp.solutions.holistic

        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.face_landmarks,
                mp_holistic.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(103, 207, 245), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1),
            )

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
