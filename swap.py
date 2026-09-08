# swap.py

import cv2
import numpy as np


class FaceSwap:
    """
    Lightweight CPU-based face replacement engine.

    Uses OpenCV Haar Cascade for face detection.
    Designed for low-memory computers.
    """

    def __init__(self, cascade_path):
        self.cascade_path = cascade_path

        self.face_detector = cv2.CascadeClassifier(
            self.cascade_path
        )

        if self.face_detector.empty():
            raise RuntimeError(
                "Could not load Haar Cascade:\n{}".format(
                    self.cascade_path
                )
            )

    def detect_faces(self, image):
        """
        Detect faces in an image.

        Returns:
            List of (x, y, w, h) face rectangles.
        """

        if image is None:
            raise ValueError("Image is empty.")

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # Improve detection without requiring GPU.
        gray = cv2.equalizeHist(gray)

        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        if faces is None:
            return []

        return list(faces)

    def _color_correct(self, source_face, target_face):
        """
        Match the replacement face's average color to the
        target face.
        """

        source_float = source_face.astype(np.float32)
        target_float = target_face.astype(np.float32)

        source_mean = np.mean(
            source_float,
            axis=(0, 1),
            keepdims=True
        )

        target_mean = np.mean(
            target_float,
            axis=(0, 1),
            keepdims=True
        )

        corrected = (
            source_float
            - source_mean
            + target_mean
        )

        corrected = np.clip(
            corrected,
            0,
            255
        )

        return corrected.astype(np.uint8)

    def _resize_face(self, face, width, height):
        """
        Resize replacement face to target face size.
        """

        return cv2.resize(
            face,
            (width, height),
            interpolation=cv2.INTER_AREA
        )

    def _create_mask(self, width, height):
        """
        Create a soft elliptical mask.

        The soft edge makes the replacement blend better
        with the surrounding image.
        """

        mask = np.zeros(
            (height, width),
            dtype=np.float32
        )

        center = (
            width // 2,
            height // 2
        )

        axes = (
            max(1, int(width * 0.43)),
            max(1, int(height * 0.48))
        )

        cv2.ellipse(
            mask,
            center,
            axes,
            0,
            0,
            360,
            1.0,
            -1
        )

        # Blur the edge for smooth blending.
        blur_size = max(
            3,
            int(min(width, height) * 0.08)
        )

        if blur_size % 2 == 0:
            blur_size += 1

        mask = cv2.GaussianBlur(
            mask,
            (blur_size, blur_size),
            0
        )

        return mask

    def _blend_face(
        self,
        target,
        replacement,
        x,
        y,
        w,
        h
    ):
        """
        Blend replacement face into target image.
        """

        if w <= 0 or h <= 0:
            return target

        target_height, target_width = target.shape[:2]

        # Keep coordinates inside image boundaries.
        x1 = max(0, x)
        y1 = max(0, y)

        x2 = min(
            target_width,
            x + w
        )

        y2 = min(
            target_height,
            y + h
        )

        if x1 >= x2 or y1 >= y2:
            return target

        actual_width = x2 - x1
        actual_height = y2 - y1

        replacement = self._resize_face(
            replacement,
            actual_width,
            actual_height
        )

        target_region = target[
            y1:y2,
            x1:x2
        ]

        replacement = self._color_correct(
            replacement,
            target_region
        )

        mask = self._create_mask(
            actual_width,
            actual_height
        )

        mask = mask[:, :, np.newaxis]

        blended = (
            replacement.astype(np.float32) * mask
            +
            target_region.astype(np.float32) * (1.0 - mask)
        )

        blended = np.clip(
            blended,
            0,
            255
        ).astype(np.uint8)

        target[
            y1:y2,
            x1:x2
        ] = blended

        return target

    def swap(self, source, replacement):
        """
        Replace the first detected face in `source`
        with the first detected face from `replacement`.

        Args:
            source:
                Target image.

            replacement:
                Image containing the replacement face.

        Returns:
            Processed BGR image.
        """

        if source is None:
            raise ValueError(
                "Source image is empty."
            )

        if replacement is None:
            raise ValueError(
                "Replacement image is empty."
            )

        source_faces = self.detect_faces(source)

        if len(source_faces) == 0:
            raise RuntimeError(
                "No face detected in source image."
            )

        replacement_faces = self.detect_faces(
            replacement
        )

        if len(replacement_faces) == 0:
            raise RuntimeError(
                "No face detected in replacement image."
            )

        # Use the largest detected face.
        source_face = max(
            source_faces,
            key=lambda face: face[2] * face[3]
        )

        replacement_face = max(
            replacement_faces,
            key=lambda face: face[2] * face[3]
        )

        sx, sy, sw, sh = source_face
        rx, ry, rw, rh = replacement_face

        replacement_crop = replacement[
            ry:ry + rh,
            rx:rx + rw
        ]

        if replacement_crop.size == 0:
            raise RuntimeError(
                "Could not extract replacement face."
            )

        # Work on a copy so the original source
        # image remains unchanged.
        result = source.copy()

        result = self._blend_face(
            result,
            replacement_crop,
            sx,
            sy,
            sw,
            sh
        )

        return result
