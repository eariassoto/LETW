from __future__ import annotations

from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from Utilities import Utilities


class HolisticProcessor:
    """
    Encapsulates the MediaPipe Holistic model lifecycle and processing.
    Used as a context manager.
    """

    def __init__(self, holistic_class: Any, confidence: float) -> None:
        self.holistic_class = holistic_class
        self.confidence = confidence
        self.model: Any | None = None

    def __enter__(self) -> HolisticProcessor:
        self.model = self.holistic_class(
            min_detection_confidence=self.confidence, min_tracking_confidence=self.confidence
        )
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.model:
            self.model.close()

    def process(self, frame: np.ndarray, draw: bool = False) -> tuple[np.ndarray, Any]:
        """Processes a single frame using the internal MediaPipe model."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.model.process(frame_rgb)
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


class MediaPipeService:
    """
    Centralizes MediaPipe Holistic initialization and detection logic.
    Provides a consistent way to process frames and draw landmarks.
    """

    def __init__(self) -> None:
        self.logger = Utilities.setup_logging()

    def start_holistic_session(self, confidence: float) -> HolisticProcessor:
        """Creates and returns a HolisticProcessor context manager."""
        return HolisticProcessor(mp.solutions.holistic.Holistic, confidence)
