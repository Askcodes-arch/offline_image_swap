import os
from PIL import Image, ImageSequence

def swap_images_in_folder(input_dir, output_dir, replacement_image_path):
    """
    Swaps/replaces static images in a folder with a target image locally.
    Keeps memory usage minimal for low-RAM systems.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    replacement_img = Image.open(replacement_image_path)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(input_dir, filename)
            try:
                with Image.open(img_path) as img:
                    # Resize replacement to match target dimensions for a clean swap layout
                    resized_replacement = replacement_img.resize(img.size)
                    output_path = os.path.join(output_dir, filename)
                    resized_replacement.save(output_path)
                    print(f"[SUCCESS] Swapped and saved: {filename}")
            except Exception as e:
                print(f"[ERROR] Could not process {filename}: {e}")

def swap_gif_frames(gif_path, replacement_frame_path, output_gif_path):
    """
    Deconstructs a GIF, swaps/overlays frames locally, and recompiles it.
    """
    base_gif = Image.open(gif_path)
    replacement = Image.open(replacement_frame_path)
    
    frames = []
    durations = []

    for frame in ImageSequence.Iterator(base_gif):
        durations.append(base_gif.info.get('duration', 100))
        current_frame = frame.convert("RGBA")
        resized_rep = replacement.resize(current_frame.size).convert("RGBA")
        
        # Simple blend or direct paste swap
        blended = Image.alpha_composite(current_frame, resized_rep)
        frames.append(blended.convert("P", palette=Image.ADAPTIVE))

    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=base_gif.info.get('loop', 0)
    )
    print(f"[SUCCESS] GIF processed successfully: {output_gif_path}")

if __name__ == "__main__":
    # Example paths configuration for local offline execution
    INPUT_FOLDER = "inputs"
    OUTPUT_FOLDER = "outputs"
    TARGET_IMAGE = "swap_target.png"
    
    print("Running offline light image/GIF swap...")
    # Uncomment below once folders and assets are set up:
    # swap_images_in_folder(INPUT_FOLDER, OUTPUT_FOLDER, TARGET_IMAGE)
