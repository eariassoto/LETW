# Developed by Anthony Villalobos 08/01/2025
# Adapted to use a video file instead of the camera
# Updated by Anthony Villalobos 23/09/2025

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
from MediaPipeService import MediaPipeService
from Utilities import Utilities


class ImageProcessor:
    """
    Converts the image from BGR to RGB, which is the format used by MediaPipe.
    Uses MediaPipe's Holistic model to process the video frames and draw landmarks.
    Returns the last frame and the results with landmarks.
    """

    def __init__(self, mp_service: MediaPipeService) -> None:
        self.service = mp_service
        self.logger = Utilities.setup_logging()

    def process_video(
        self,
        video_path: str,
        confidence: float,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> tuple[np.ndarray | None, Any | None]:  # Loads the video, processes it and draws the landmarks
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"No se pudo abrir el vídeo: {video_path}")
            self.logger.error(f"No se pudo abrir el vídeo: {video_path}")
            return None, None

        last_frame, last_result = None, None

        with self.service.start_holistic_session(confidence) as processor:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if transform:
                    frame = transform(frame)

                image, results = processor.process(frame, draw=True)

                if (
                    results.pose_landmarks
                    or results.face_landmarks
                    or results.left_hand_landmarks
                    or results.right_hand_landmarks
                ):
                    last_frame = frame
                    last_result = results

                # Remove the comment to show the video with the landmarks; used during development, not needed now
                cv2.imshow("Video Detection", image)

                cap.read()
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()
        return last_frame, last_result
