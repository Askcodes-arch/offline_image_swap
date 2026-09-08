# main.py

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from swap import FaceSwap


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(BASE_DIR, "models")

CASCADE = os.path.join(
    MODEL_DIR,
    "haarcascade_frontalface_default.xml"
)


SUPPORTED_IMAGES = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


class OfflineImageSwapApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Offline Image Swap")
        self.root.geometry("900x650")
        self.root.minsize(760, 560)

        self.source_path = ""
        self.replacement_path = ""
        self.multi_paths = []

        self.source_preview = None
        self.replacement_preview = None

        self.engine = None

        os.makedirs(INPUT_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)

        self.create_interface()

    # ---------------------------------------------------------
    # INTERFACE
    # ---------------------------------------------------------

    def create_interface(self):

        title = tk.Label(
            self.root,
            text="Offline Image Swap",
            font=("Arial", 20, "bold")
        )
        title.pack(pady=(12, 4))

        subtitle = tk.Label(
            self.root,
            text="CPU mode • Offline • Lightweight",
            font=("Arial", 10)
        )
        subtitle.pack(pady=(0, 10))

        notebook = ttk.Notebook(self.root)
        notebook.pack(
            fill="both",
            expand=True,
            padx=12,
            pady=8
        )

        self.single_tab = ttk.Frame(notebook)
        self.multi_tab = ttk.Frame(notebook)
        self.gif_tab = ttk.Frame(notebook)

        notebook.add(
            self.single_tab,
            text="  Image Swap  "
        )

        notebook.add(
            self.multi_tab,
            text="  Multiple Images  "
        )

        notebook.add(
            self.gif_tab,
            text="  GIF  "
        )

        self.create_single_tab()
        self.create_multi_tab()
        self.create_gif_tab()

        self.status_var = tk.StringVar(
            value="Ready."
        )

        status = tk.Label(
            self.root,
            textvariable=self.status_var,
            anchor="w",
            relief="sunken",
            padx=8
        )
        status.pack(
            fill="x",
            side="bottom"
        )

    # ---------------------------------------------------------
    # SINGLE IMAGE TAB
    # ---------------------------------------------------------

    def create_single_tab(self):

        controls = ttk.Frame(
            self.single_tab
        )
        controls.pack(
            fill="x",
            padx=15,
            pady=15
        )

        ttk.Button(
            controls,
            text="Select Target Image",
            command=self.select_source
        ).grid(
            row=0,
            column=0,
            padx=5,
            pady=5
        )

        ttk.Button(
            controls,
            text="Select Replacement Face",
            command=self.select_replacement
        ).grid(
            row=0,
            column=1,
            padx=5,
            pady=5
        )

        ttk.Button(
            controls,
            text="Swap Face",
            command=self.start_swap
        ).grid(
            row=0,
            column=2,
            padx=5,
            pady=5
        )

        controls.columnconfigure(
            3,
            weight=1
        )

        self.source_label = ttk.Label(
            self.single_tab,
            text="Target: No image selected"
        )
        self.source_label.pack(
            anchor="w",
            padx=20,
            pady=5
        )

        self.replacement_label = ttk.Label(
            self.single_tab,
            text="Replacement: No image selected"
        )
        self.replacement_label.pack(
            anchor="w",
            padx=20,
            pady=5
        )

        preview_frame = ttk.Frame(
            self.single_tab
        )
        preview_frame.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=10
        )

        left = ttk.LabelFrame(
            preview_frame,
            text="Target"
        )
        left.pack(
            side="left",
            fill="both",
            expand=True,
            padx=5
        )

        right = ttk.LabelFrame(
            preview_frame,
            text="Replacement"
        )
        right.pack(
            side="right",
            fill="both",
            expand=True,
            padx=5
        )

        self.source_preview_label = tk.Label(
            left,
            text="No image"
        )
        self.source_preview_label.pack(
            fill="both",
            expand=True
        )

        self.replacement_preview_label = tk.Label(
            right,
            text="No image"
        )
        self.replacement_preview_label.pack(
            fill="both",
            expand=True
        )

    # ---------------------------------------------------------
    # MULTIPLE IMAGE TAB
    # ---------------------------------------------------------

    def create_multi_tab(self):

        top = ttk.Frame(
            self.multi_tab
        )
        top.pack(
            fill="x",
            padx=15,
            pady=15
        )

        ttk.Button(
            top,
            text="Add Images",
            command=self.add_multiple_images
        ).pack(
            side="left",
            padx=5
        )

        ttk.Button(
            top,
            text="Clear",
            command=self.clear_multiple_images
        ).pack(
            side="left",
            padx=5
        )

        self.multi_list = tk.Listbox(
            self.multi_tab,
            selectmode=tk.EXTENDED
        )
        self.multi_list.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=10
        )

        bottom = ttk.Frame(
            self.multi_tab
        )
        bottom.pack(
            fill="x",
            padx=15,
            pady=10
        )

        ttk.Label(
            bottom,
            text="Frame delay (ms):"
        ).pack(
            side="left"
        )

        self.multi_delay_var = tk.IntVar(
            value=500
        )

        ttk.Spinbox(
            bottom,
            from_=50,
            to=10000,
            increment=50,
            textvariable=self.multi_delay_var,
            width=8
        ).pack(
            side="left",
            padx=8
        )

        ttk.Button(
            bottom,
            text="Create GIF",
            command=self.create_multi_gif
        ).pack(
            side="right",
            padx=5
        )

    # ---------------------------------------------------------
    # GIF TAB
    # ---------------------------------------------------------

    def create_gif_tab(self):

        frame = ttk.Frame(
            self.gif_tab
        )
        frame.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=20
        )

        ttk.Label(
            frame,
            text="Create an animated GIF from multiple images.",
            font=("Arial", 12)
        ).pack(
            pady=10
        )

        ttk.Button(
            frame,
            text="Select Images",
            command=self.add_multiple_images
        ).pack(
            pady=8
        )

        ttk.Label(
            frame,
            text="Frame delay (milliseconds)"
        ).pack(
            pady=(20, 4)
        )

        self.gif_delay_var = tk.IntVar(
            value=500
        )

        ttk.Spinbox(
            frame,
            from_=50,
            to=10000,
            increment=50,
            textvariable=self.gif_delay_var,
            width=10
        ).pack()

        ttk.Button(
            frame,
            text="Create GIF",
            command=self.create_multi_gif
        ).pack(
            pady=20
        )

        self.gif_info = ttk.Label(
            frame,
            text="No images selected."
        )
        self.gif_info.pack(
            pady=10
        )

    # ---------------------------------------------------------
    # FILE SELECTION
    # ---------------------------------------------------------

    def select_source(self):

        path = filedialog.askopenfilename(
            title="Select target image",
            filetypes=[
                (
                    "Image files",
                    "*.jpg *.jpeg *.png *.bmp *.webp"
                ),
                ("All files", "*.*")
            ]
        )

        if not path:
            return

        self.source_path = path

        self.source_label.config(
            text="Target: {}".format(path)
        )

        self.show_preview(
            path,
            self.source_preview_label,
            "source"
        )

        self.status_var.set(
            "Target image selected."
        )

    def select_replacement(self):

        path = filedialog.askopenfilename(
            title="Select replacement face",
            filetypes=[
                (
                    "Image files",
                    "*.jpg *.jpeg *.png *.bmp *.webp"
                ),
                ("All files", "*.*")
            ]
        )

        if not path:
            return

        self.replacement_path = path

        self.replacement_label.config(
            text="Replacement: {}".format(path)
        )

        self.show_preview(
            path,
            self.replacement_preview_label,
            "replacement"
        )

        self.status_var.set(
            "Replacement image selected."
        )

    def add_multiple_images(self):

        paths = filedialog.askopenfilenames(
            title="Select images",
            filetypes=[
                (
                    "Image files",
                    "*.jpg *.jpeg *.png *.bmp *.webp"
                ),
                ("All files", "*.*")
            ]
        )

        if not paths:
            return

        for path in paths:

            if path not in self.multi_paths:
                self.multi_paths.append(path)
                self.multi_list.insert(
                    tk.END,
                    os.path.basename(path)
                )

        self.gif_info.config(
            text="{} image(s) selected.".format(
                len(self.multi_paths)
            )
        )

        self.status_var.set(
            "{} image(s) selected.".format(
                len(self.multi_paths)
            )
        )

    def clear_multiple_images(self):

        self.multi_paths = []

        self.multi_list.delete(
            0,
            tk.END
        )

        self.gif_info.config(
            text="No images selected."
        )

        self.status_var.set(
            "Image list cleared."
        )

    # ---------------------------------------------------------
    # PREVIEW
    # ---------------------------------------------------------

    def show_preview(
        self,
        path,
        label,
        preview_type
    ):

        try:

            image = Image.open(path)

            image.thumbnail(
                (360, 360),
                Image.LANCZOS
            )

            photo = ImageTk.PhotoImage(
                image
            )

            label.config(
                image=photo,
                text=""
            )

            label.image = photo

            if preview_type == "source":
                self.source_preview = photo
            else:
                self.replacement_preview = photo

        except Exception as error:

            label.config(
                image="",
                text="Preview unavailable"
            )

            self.status_var.set(
                "Preview error: {}".format(error)
            )

    # ---------------------------------------------------------
    # ENGINE
    # ---------------------------------------------------------

    def get_engine(self):

        if not os.path.exists(CASCADE):
            raise RuntimeError(
                "Face detector not found:\n{}".format(
                    CASCADE
                )
            )

        if self.engine is None:
            self.engine = FaceSwap(
                CASCADE
            )

        return self.engine

    # ---------------------------------------------------------
    # IMAGE SWAP
    # ---------------------------------------------------------

    def start_swap(self):

        if not self.source_path:
            messagebox.showwarning(
                "Missing image",
                "Please select the target image."
            )
            return

        if not self.replacement_path:
            messagebox.showwarning(
                "Missing image",
                "Please select the replacement image."
            )
            return

        self.status_var.set(
            "Processing..."
        )

        thread = threading.Thread(
            target=self.perform_swap,
            daemon=True
        )

        thread.start()

    def perform_swap(self):

        try:

            source = cv2.imread(
                self.source_path
            )

            replacement = cv2.imread(
                self.replacement_path
            )

            if source is None:
                raise RuntimeError(
                    "Could not read target image."
                )

            if replacement is None:
                raise RuntimeError(
                    "Could not read replacement image."
                )

            engine = self.get_engine()

            result = engine.swap(
                source,
                replacement
            )

            output_path = os.path.join(
                OUTPUT_DIR,
                "result.jpg"
            )

            success = cv2.imwrite(
                output_path,
                result,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    95
                ]
            )

            if not success:
                raise RuntimeError(
                    "Could not save output image."
                )

            self.root.after(
                0,
                lambda: self.swap_finished(
                    output_path
                )
            )

        except Exception as error:

            self.root.after(
                0,
                lambda: self.show_error(
                    str(error)
                )
            )

    def swap_finished(self, output_path):

        self.status_var.set(
            "Swap complete."
        )

        messagebox.showinfo(
            "Success",
            "Image saved to:\n\n{}".format(
                output_path
            )
        )

    # ---------------------------------------------------------
    # GIF CREATION
    # ---------------------------------------------------------

    def create_multi_gif(self):

        if len(self.multi_paths) < 2:

            messagebox.showwarning(
                "Not enough images",
                "Select at least 2 images."
            )

            return

        try:

            delay = int(
                self.gif_delay_var.get()
            )

        except Exception:

            delay = 500

        self.status_var.set(
            "Creating GIF..."
        )

        thread = threading.Thread(
            target=self.perform_gif,
            args=(delay,),
            daemon=True
        )

        thread.start()

    def perform_gif(self, delay):

        frames = []

        try:

            for path in self.multi_paths:

                image = Image.open(path)

                image = image.convert(
                    "RGB"
                )

                # Keep memory usage reasonable.
                image.thumbnail(
                    (1280, 1280),
                    Image.LANCZOS
                )

                frames.append(
                    image.copy()
                )

                image.close()

            if not frames:
                raise RuntimeError(
                    "No usable images found."
                )

            # Make all frames the same size.
            width = max(
                frame.width
                for frame in frames
            )

            height = max(
                frame.height
                for frame in frames
            )

            normalized = []

            for frame in frames:

                canvas = Image.new(
                    "RGB",
                    (width, height),
                    "black"
                )

                x = (
                    width - frame.width
                ) // 2

                y = (
                    height - frame.height
                ) // 2

                canvas.paste(
                    frame,
                    (x, y)
                )

                normalized.append(
                    canvas
                )

            output_path = os.path.join(
                OUTPUT_DIR,
                "animation.gif"
            )

            normalized[0].save(
                output_path,
                save_all=True,
                append_images=normalized[1:],
                duration=delay,
                loop=0,
                optimize=True
            )

            for frame in frames:
                frame.close()

            for frame in normalized:
                frame.close()

            self.root.after(
                0,
                lambda: self.gif_finished(
                    output_path
                )
            )

        except Exception as error:

            for frame in frames:
                try:
                    frame.close()
                except Exception:
                    pass

            self.root.after(
                0,
                lambda: self.show_error(
                    str(error)
                )
            )

    def gif_finished(self, output_path):

        self.status_var.set(
            "GIF created."
        )

        messagebox.showinfo(
            "GIF complete",
            "GIF saved to:\n\n{}".format(
                output_path
            )
        )

    # ---------------------------------------------------------
    # ERROR
    # ---------------------------------------------------------

    def show_error(self, error):

        self.status_var.set(
            "Error."
        )

        messagebox.showerror(
            "Error",
            error
        )


def main():

    root = tk.Tk()

    app = OfflineImageSwapApp(
        root
    )

    root.mainloop()


if __name__ == "__main__":
    main()
