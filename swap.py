# swap.py

import cv2
import numpy as np
from PIL import Image


class FaceSwap:
    """
    Lightweight CPU-based face replacement engine.

    Uses OpenCV Haar Cascade.
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

    # =========================================================
    # FACE DETECTION
    # =========================================================

    def detect_faces(self, image):
        """Detect faces and return (x, y, w, h)."""

        if image is None:
            raise ValueError(
                "Image is empty."
            )

        if len(image.shape) == 2:
            gray = image
        else:
            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

        gray = cv2.equalizeHist(
            gray
        )

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

    # =========================================================
    # FACE CROP
    # =========================================================

    def _crop_face(
        self,
        image,
        face
    ):
        """Extract a face from an image."""

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

    # =========================================================
    # RESIZE
    # =========================================================

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

    # =========================================================
    # COLOR CORRECTION
    # =========================================================

    def _color_correct(
        self,
        source_face,
        target_face
    ):
        """
        Basic color/brightness matching.
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

    # =========================================================
    # MASK
    # =========================================================

    def _create_mask(
        self,
        width,
        height
    ):
        """
        Create a soft elliptical mask.
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
            max(
                1,
                int(width * 0.42)
            ),
            max(
                1,
                int(height * 0.47)
            )
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
            int(
                min(width, height)
                * 0.08
            )
        )

        if blur % 2 == 0:
            blur += 1

        mask = cv2.GaussianBlur(
            mask,
            (blur, blur),
            0
        )

        return mask

    # =========================================================
    # BLEND
    # =========================================================

    def _blend_face(
        self,
        target,
        replacement,
        face
    ):
        """
        Blend a replacement face into target.
        """

        x, y, w, h = face

        target_height, target_width = (
            target.shape[:2]
        )

        x1 = max(
            0,
            x
        )

        y1 = max(
            0,
            y
        )

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

    # =========================================================
    # SINGLE FACE SWAP
    # =========================================================

    def swap(
        self,
        target,
        replacement
    ):
        """
        Replace the largest face in target.
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

        return self._blend_face(
            result,
            replacement_crop,
            target_face
        )

    # =========================================================
    # ALL FACE SWAP
    # =========================================================

    def swap_all_faces(
        self,
        target,
        replacement
    ):
        """
        Replace every detected target face
        with the same replacement face.
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

    # =========================================================
    # PIL / OPENCV CONVERSION
    # =========================================================

    @staticmethod
    def pil_to_cv(image):
        """PIL RGB -> OpenCV BGR."""

        rgb = np.array(
            image.convert("RGB")
        )

        return cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2BGR
        )

    @staticmethod
    def cv_to_pil(image):
        """OpenCV BGR -> PIL RGB."""

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        return Image.fromarray(
            rgb
        )

    # =========================================================
    # GIF FRAME PROCESSING
    # =========================================================

    def process_gif_frame(
        self,
        frame,
        replacement,
        replace_all=False
    ):
        """
        Process one GIF frame.
        """

        cv_frame = self.pil_to_cv(
            frame
        )

        try:

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

        finally:

            del cv_frame

    # =========================================================
    # GIF FACE SWAP
    # =========================================================

    def swap_gif(
        self,
        gif_path,
        replacement_path,
        output_path,
        replace_all=False,
        max_width=960,
        max_frames=300,
        progress_callback=None
    ):
        """
        Process an animated GIF frame by frame.

        max_width:
            Maximum output width.

        max_frames:
            Maximum number of frames.

        progress_callback:
            Function receiving current,total.
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
            del replacement

            raise RuntimeError(
                "No face detected in replacement image."
            )

        gif = Image.open(
            gif_path
        )

        processed_frames = []

        try:

            total_frames = getattr(
                gif,
                "n_frames",
                1
            )

            if total_frames > max_frames:

                raise RuntimeError(
                    "GIF contains {} frames.\n\n"
                    "Maximum allowed: {} frames."
                    .format(
                        total_frames,
                        max_frames
                    )
                )

            duration = gif.info.get(
                "duration",
                500
            )

            loop = gif.info.get(
                "loop",
                0
            )

            for index in range(
                total_frames
            ):

                gif.seek(
                    index
                )

                frame = gif.convert(
                    "RGB"
                )

                try:

                    if frame.width > max_width:

                        new_height = int(
                            frame.height
                            * max_width
                            / frame.width
                        )

                        resized = frame.resize(
                            (
                                max_width,
                                new_height
                            ),
                            Image.LANCZOS
                        )

                        frame.close()

                        frame = resized

                    processed = (
                        self.process_gif_frame(
                            frame,
                            replacement,
                            replace_all
                        )
                    )

                    processed_frames.append(
                        processed
                    )

                    if progress_callback:

                        progress_callback(
                            index + 1,
                            total_frames
                        )

                finally:

                    try:
                        frame.close()
                    except Exception:
                        pass

            if not processed_frames:

                raise RuntimeError(
                    "No usable GIF frames."
                )

            first = processed_frames[0]

            first.save(
                output_path,
                save_all=True,
                append_images=(
                    processed_frames[1:]
                ),
                duration=duration,
                loop=loop,
                optimize=True
            )

        finally:

            gif.close()

            for frame in processed_frames:

                try:
                    frame.close()
                except Exception:
                    pass

            del replacement

    # =========================================================
    # CREATE GIF FROM IMAGES
    # =========================================================

    def create_gif(
        self,
        image_paths,
        output_path,
        delay=500,
        max_width=960
    ):
        """
        Create an animated GIF from images.
        """

        if not image_paths:

            raise RuntimeError(
                "No images supplied."
            )

        frames = []

        try:

            for path in image_paths:

                image = Image.open(
                    path
                )

                try:

                    image = image.convert(
                        "RGB"
                    )

                    if image.width > max_width:

                        new_height = int(
                            image.height
                            * max_width
                            / image.width
                        )

                        resized = image.resize(
                            (
                                max_width,
                                new_height
                            ),
                            Image.LANCZOS
                        )

                        image.close()

                        image = resized

                    frames.append(
                        image.copy()
                    )

                finally:

                    try:
                        image.close()
                    except Exception:
                        pass

            if not frames:

                raise RuntimeError(
                    "No usable images."
                )

            width = max(
                frame.width
                for frame in frames
            )

            height = max(
                frame.height
                for frame in frames
            )

            normalized = []

            try:

                for frame in frames:

                    canvas = Image.new(
                        "RGB",
                        (
                            width,
                            height
                        ),
                        "black"
                    )

                    x = (
                        width
                        - frame.width
                    ) // 2

                    y = (
                        height
                        - frame.height
                    ) // 2

                    canvas.paste(
                        frame,
                        (x, y)
                    )

                    normalized.append(
                        canvas
                    )

                normalized[0].save(
                    output_path,
                    save_all=True,
                    append_images=(
                        normalized[1:]
                    ),
                    duration=delay,
                    loop=0,
                    optimize=True
                )

            finally:

                for frame in normalized:

                    try:
                        frame.close()
                    except Exception:
                        pass

        finally:

            for frame in frames:

                try:
                    frame.close()
                except Exception:
                    pass
