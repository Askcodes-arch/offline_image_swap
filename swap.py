# swap.py

import cv2
import numpy as np
from PIL import Image


class FaceSwap:
    """
    Lightweight CPU face replacement engine.

    Uses OpenCV Haar Cascade only.
    No Internet, CUDA, GPU, or large AI model required.
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

        if image is None:
            raise ValueError(
                "Image is empty."
            )

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

        if not faces:
            return None

        return max(
            faces,
            key=lambda f: f[2] * f[3]
        )

    # =========================================================
    # FACE CROP
    # =========================================================

    def _crop_face(self, image, face):

        x, y, w, h = face

        # Expand crop slightly.
        # This gives the blending area some surrounding skin.
        expand_x = int(w * 0.18)
        expand_y = int(h * 0.20)

        x1 = max(
            0,
            x - expand_x
        )

        y1 = max(
            0,
            y - expand_y
        )

        x2 = min(
            image.shape[1],
            x + w + expand_x
        )

        y2 = min(
            image.shape[0],
            y + h + expand_y
        )

        crop = image[
            y1:y2,
            x1:x2
        ]

        if crop.size == 0:
            raise RuntimeError(
                "Could not extract face."
            )

        return crop.copy()

    # =========================================================
    # COLOR MATCHING
    # =========================================================

    def _match_color(
        self,
        source,
        target
    ):
        """
        Match source brightness/color statistics
        to the target region.
        """

        source_float = source.astype(
            np.float32
        )

        target_float = target.astype(
            np.float32
        )

        source_mean = np.mean(
            source_float,
            axis=(0, 1)
        )

        target_mean = np.mean(
            target_float,
            axis=(0, 1)
        )

        source_std = np.std(
            source_float,
            axis=(0, 1)
        )

        target_std = np.std(
            target_float,
            axis=(0, 1)
        )

        source_std[
            source_std < 1.0
        ] = 1.0

        result = (
            (
                source_float
                - source_mean
            )
            *
            (
                target_std
                / source_std
            )
        ) + target_mean

        return np.clip(
            result,
            0,
            255
        ).astype(
            np.uint8
        )

    # =========================================================
    # SOFT FACE MASK
    # =========================================================

    def _create_mask(
        self,
        width,
        height
    ):

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
                int(width * 0.40)
            ),
            max(
                1,
                int(height * 0.45)
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

        blur_size = int(
            min(width, height) * 0.10
        )

        blur_size = max(
            5,
            blur_size
        )

        if blur_size % 2 == 0:
            blur_size += 1

        mask = cv2.GaussianBlur(
            mask,
            (
                blur_size,
                blur_size
            ),
            0
        )

        # Keep the outer edges very soft.
        mask = np.clip(
            mask,
            0.0,
            1.0
        )

        return mask

    # =========================================================
    # ALIGNED FACE
    # =========================================================

    def _prepare_face(
        self,
        source_face,
        width,
        height
    ):

        return cv2.resize(
            source_face,
            (
                width,
                height
            ),
            interpolation=cv2.INTER_AREA
        )

    # =========================================================
    # BLEND
    # =========================================================

    def _blend_face(
        self,
        target,
        replacement,
        face
    ):

        x, y, w, h = face

        target_height = target.shape[0]
        target_width = target.shape[1]

        # Expand destination area.
        expand_x = int(w * 0.18)
        expand_y = int(h * 0.20)

        x1 = max(
            0,
            x - expand_x
        )

        y1 = max(
            0,
            y - expand_y
        )

        x2 = min(
            target_width,
            x + w + expand_x
        )

        y2 = min(
            target_height,
            y + h + expand_y
        )

        if x1 >= x2 or y1 >= y2:
            return target

        width = x2 - x1
        height = y2 - y1

        replacement = self._prepare_face(
            replacement,
            width,
            height
        )

        target_region = target[
            y1:y2,
            x1:x2
        ]

        replacement = self._match_color(
            replacement,
            target_region
        )

        mask = self._create_mask(
            width,
            height
        )

        # Slightly stronger center.
        mask = np.power(
            mask,
            0.85
        )

        mask = mask[
            :,
            :,
            np.newaxis
        ]

        source_float = replacement.astype(
            np.float32
        )

        target_float = target_region.astype(
            np.float32
        )

        blended = (
            source_float * mask
            +
            target_float * (
                1.0 - mask
            )
        )

        target[
            y1:y2,
            x1:x2
        ] = np.clip(
            blended,
            0,
            255
        ).astype(
            np.uint8
        )

        return target

    # =========================================================
    # SINGLE FACE SWAP
    # =========================================================

    def swap(
        self,
        target,
        replacement
    ):

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

    # =========================================================
    # ALL FACES
    # =========================================================

    def swap_all_faces(
        self,
        target,
        replacement
    ):

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
    # PIL / OPENCV
    # =========================================================

    @staticmethod
    def pil_to_cv(image):

        rgb = np.array(
            image.convert("RGB")
        )

        return cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2BGR
        )

    @staticmethod
    def cv_to_pil(image):

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        return Image.fromarray(
            rgb
        )

    # =========================================================
    # GIF FRAME
    # =========================================================

    def process_gif_frame(
        self,
        frame,
        replacement,
        replace_all=False
    ):

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
                    "GIF has {} frames. "
                    "Maximum is {} frames."
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
                    "No GIF frames were processed."
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
    # CREATE GIF FROM MULTIPLE IMAGES
    # =========================================================

    def create_gif(
        self,
        image_paths,
        output_path,
        delay=500,
        max_width=960
    ):

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
