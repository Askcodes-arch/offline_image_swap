# main.py

import os
import cv2

from swap import FaceSwap


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(BASE_DIR, "models")

SOURCE_IMAGE = os.path.join(INPUT_DIR, "source.jpg")
REPLACEMENT_IMAGE = os.path.join(INPUT_DIR, "replacement.jpg")

CASCADE = os.path.join(
    MODEL_DIR,
    "haarcascade_frontalface_default.xml"
)

OUTPUT_IMAGE = os.path.join(
    OUTPUT_DIR,
    "result.jpg"
)


def main():

    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("================================")
    print(" Offline Image Swap")
    print(" CPU mode")
    print("================================")

    if not os.path.exists(SOURCE_IMAGE):
        print("ERROR: source.jpg not found.")
        return

    if not os.path.exists(REPLACEMENT_IMAGE):
        print("ERROR: replacement.jpg not found.")
        return

    if not os.path.exists(CASCADE):
        print("ERROR: face detector not found:")
        print(CASCADE)
        return

    source = cv2.imread(SOURCE_IMAGE)
    replacement = cv2.imread(REPLACEMENT_IMAGE)

    if source is None:
        print("ERROR: Could not read source.jpg")
        return

    if replacement is None:
        print("ERROR: Could not read replacement.jpg")
        return

    print("Detecting faces...")

    try:
        engine = FaceSwap(CASCADE)

        result = engine.swap(
            source,
            replacement
        )

    except Exception as error:
        print("ERROR:", error)
        return

    cv2.imwrite(
        OUTPUT_IMAGE,
        result,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )

    print()
    print("SUCCESS!")
    print("Output:")
    print(OUTPUT_IMAGE)


if __name__ == "__main__":
    main()
