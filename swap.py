# swap.py

import cv2
import numpy as np
from PIL import Image


class FaceSwap:
    """
    Lightweight CPU face-swap engine.

    Uses OpenCV Haar Cascade for face detection.
    No AI model, GPU, CUDA, or Internet required.
    """

    def __init__(self, cascade_path):
        self.cascade_path = cascade_path

        self.face_detector = cv2.CascadeClassifier(
            cascade_path
        )

        if self.face_detector.empty():
            raise RuntimeError(
                "Could not load face detector:\n{}".format(
                    cascade_path
                )
            )

    # ---------------------------------------------------------
    # FACE DETECTION
    # ---------------------------------------------------------

    def detect_faces(self, image):
        """Return detected faces as (x, y, w, h)."""

        if image is None:
            raise ValueError("Image is empty.")

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.equalizeHist(gray)

        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        if faces is None:
            return []

        return list(faces)

    def largest_face(self, faces):
        """Return the largest detected face."""

        if not faces:
            return None

        return max(
            faces,
            key=lambda f: f[2] * f[3]
        )

    # ---------------------------------------------------------
    # FACE PREPARATION
    # ---------------------------------------------------------

    def _crop_face(self, image, face):
        """Crop a detected face."""

        x, y, w, h = face

        crop = image[
            y:y + h,
            x:x + w
        ]

        if crop.size == 0:
            raise RuntimeError(
                "Could not extract face."
            )

        return crop.copy()

    def _resize_face(
        self,
        face,
        width,
        height
    ):
        return cv2.resize(
            face,
            (width, height),
            interpolation=cv2.INTER_AREA
        )

    # ---------------------------------------------------------
    # COLOR CORRECTION
    # ---------------------------------------------------------

    def _color_correct(
        self,
        source_face,
        target_face
    ):
        """
        Match average color of replacement face
        to the target area.
        """

        source_float = source_face.astype(
            np.float32
        )

        target_float = target_face.astype(
            np.float32
        )

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

        result = (
            source_float
            - source_mean
            + target_mean
        )

        return np.clip(
            result,
            0,
            255
        ).astype(np.uint8)

    # ---------------------------------------------------------
    # MASK
    # ---------------------------------------------------------

    def _create_mask(
        self,
        width,
        height
    ):
        """
        Create a soft elliptical face mask.
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
            max(1, int(width * 0.42)),
            max(1, int(height * 0.47))
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

        blur = max(
            3,
            int(min(width, height) * 0.08)
        )

        if blur % 2 == 0:
            blur += 1

        mask = cv2.GaussianBlur(
            mask,
            (blur, blur),
            0
        )

        return mask

    # ---------------------------------------------------------
    # SINGLE FACE BLEND
    # ---------------------------------------------------------

    def _blend_face(
        self,
        target,
        replacement,
        face
    ):
        """
        Blend replacement face into target.
        """

        x, y, w, h = face

        target_height, target_width = (
            target.shape[:2]
        )

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

        width = x2 - x1
        height = y2 - y1

        replacement = self._resize_face(
            replacement,
            width,
            height
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
            width,
            height
        )

        mask = mask[:, :, np.newaxis]

        result = (
            replacement.astype(
                np.float32
            ) * mask
            +
            target_region.astype(
                np.float32
            ) * (1.0 - mask)
        )

        target[
            y1:y2,
            x1:x2
        ] = np.clip(
            result,
            0,
            255
        ).astype(np.uint8)

        return target

    # ---------------------------------------------------------
    # SINGLE IMAGE SWAP
    # ---------------------------------------------------------

    def swap(
        self,
        target,
        replacement
    ):
        """
        Replace the largest face in target
        with the largest face in replacement.
        """

        if target is None:
            raise ValueError(
                "Target image is empty."
            )

        if replacement is None:
            raise ValueError(
                "Replacement image is empty."
            )

        target_faces = self.detect_faces(
            target
        )

        replacement_faces = self.detect_faces(
            replacement
        )

        if not target_faces:
            raise RuntimeError(
                "No face detected in target image."
            )

        if not replacement_faces:
            raise RuntimeError(
                "No face detected in replacement image."
            )

        target_face = self.largest_face(
            target_faces
        )

        replacement_face = self.largest_face(
            replacement_faces
        )

        replacement_crop = self._crop_face(
            replacement,
            replacement_face
        )

        result = target.copy()

        result = self._blend_face(
            result,
            replacement_crop,
            target_face
        )

        return result

    # ---------------------------------------------------------
    # MULTIPLE FACES
    # ---------------------------------------------------------

    def swap_all_faces(
        self,
        target,
        replacement
    ):
        """
        Replace every detected target face with
        the same replacement face.
        """

        target_faces = self.detect_faces(
            target
        )

        replacement_faces = self.detect_faces(
            replacement
        )

        if not target_faces:
            raise RuntimeError(
                "No face detected in target image."
            )

        if not replacement_faces:
            raise RuntimeError(
                "No face detected in replacement image."
            )

        replacement_face = self.largest_face(
            replacement_faces
        )

        replacement_crop = self._crop_face(
            replacement,
            replacement_face
        )

        result = target.copy()

        for face in target_faces:

            result = self._blend_face(
                result,
                replacement_crop,
                face
            )

        return result

    # ---------------------------------------------------------
    # GIF FRAME CONVERSION
    # ---------------------------------------------------------

    @staticmethod
    def pil_to_cv(image):
        """
        PIL RGB image -> OpenCV BGR image.
        """

        rgb = np.array(
            image.convert("RGB")
        )

        return cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2BGR
        )

    @staticmethod
    def cv_to_pil(image):
        """
        OpenCV BGR image -> PIL RGB image.
        """

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        return Image.fromarray(
            rgb
        )

    # ---------------------------------------------------------
    # GIF FACE SWAP
    # ---------------------------------------------------------

    def process_gif_frame(
        self,
        frame,
        replacement,
        replace_all=False
    ):
        """
        Process one GIF frame.

        Args:
            frame:
                PIL RGB frame.

            replacement:
                OpenCV BGR replacement image.

            replace_all:
                Replace all detected faces if True.

        Returns:
            PIL RGB processed frame.
        """

        cv_frame = self.pil_to_cv(
            frame
        )

        if replace_all:

            result = self.swap_all_faces(
                cv_frame,
                replacement
            )

        else:

            result = self.swap(
                cv_frame,
                replacement
            )

        return self.cv_to_pil(
            result
        )

    def swap_gif(
        self,
        gif_path,
        replacement_path,
        output_path,
        replace_all=False,
        max_width=1280
    ):
        """
        Face-swap an animated GIF frame by frame.

        The GIF is processed sequentially to reduce
        memory usage.
        """

        replacement = cv2.imread(
            replacement_path
        )

        if replacement is None:
            raise RuntimeError(
                "Could not read replacement image."
            )

        replacement_faces = self.detect_faces(
            replacement
        )

        if not replacement_faces:
            raise RuntimeError(
                "No face detected in replacement image."
            )

        gif = Image.open(
            gif_path
        )

        processed_frames = []

        try:

            frame_count = getattr(
                gif,
                "n_frames",
                1
            )

            for index in range(frame_count):

                gif.seek(index)

                frame = gif.convert(
                    "RGB"
                )

                # Reduce very large GIF frames.
                if frame.width > max_width:

                    new_height = int(
                        frame.height
                        * max_width
                        / frame.width
                    )

                    frame = frame.resize(
                        (
                            max_width,
                            new_height
                        ),
                        Image.LANCZOS
                    )

                try:

                    processed = (
                        self.process_gif_frame(
                            frame,
                            replacement,
                            replace_all
                        )
                    )

                    processed_frames.append(
                        processed.copy()
                    )

                finally:

                    frame.close()

            if not processed_frames:
                raise RuntimeError(
                    "GIF contains no usable frames."
                )

            first = processed_frames[0]

            first.save(
                output_path,
                save_all=True,
                append_images=(
                    processed_frames[1:]
                ),
                duration=gif.info.get(
                    "duration",
                    500
                ),
                loop=gif.info.get(
                    "loop",
                    0
                ),
                optimize=True
            )

        finally:

            gif.close()

            for frame in processed_frames:
                try:
                    frame.close()
                except Exception:
                    pass
