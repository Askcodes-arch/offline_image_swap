# swap.py

import cv2
import numpy as np


class FaceSwap:
    def __init__(self, cascade_path):
        self.detector = cv2.CascadeClassifier(cascade_path)

        if self.detector.empty():
            raise RuntimeError(
                "Could not load face detector: " + cascade_path
            )

    def detect_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        if len(faces) == 0:
            return None

        # Select largest detected face
        faces = sorted(
            faces,
            key=lambda f: f[2] * f[3],
            reverse=True
        )

        return faces[0]

    def swap(self, source, replacement):
        source_face = self.detect_face(source)
        replacement_face = self.detect_face(replacement)

        if source_face is None:
            raise RuntimeError("No face found in source image.")

        if replacement_face is None:
            raise RuntimeError("No face found in replacement image.")

        sx, sy, sw, sh = source_face
        rx, ry, rw, rh = replacement_face

        # Crop replacement face
        face = replacement[
            ry:ry + rh,
            rx:rx + rw
        ]

        if face.size == 0:
            raise RuntimeError("Invalid replacement face.")

        # Resize replacement face to target face size
        face = cv2.resize(
            face,
            (sw, sh),
            interpolation=cv2.INTER_AREA
        )

        # Create elliptical mask
        mask = np.zeros((sh, sw), dtype=np.uint8)

        center = (
            sw // 2,
            sh // 2
        )

        axes = (
            max(1, int(sw * 0.45)),
            max(1, int(sh * 0.48))
        )

        cv2.ellipse(
            mask,
            center,
            axes,
            0,
            0,
            360,
            255,
            -1
        )

        # Slightly soften mask edges
        mask = cv2.GaussianBlur(
            mask,
            (15, 15),
            0
        )

        # Put replacement into temporary image
        result = source.copy()

        roi = result[
            sy:sy + sh,
            sx:sx + sw
        ]

        # Alpha blend
        alpha = mask.astype(np.float32) / 255.0
        alpha = alpha[:, :, np.newaxis]

        blended = (
            face.astype(np.float32) * alpha +
            roi.astype(np.float32) * (1.0 - alpha)
        )

        result[
            sy:sy + sh,
            sx:sx + sw
        ] = blended.astype(np.uint8)

        return result

