"""Capture original frames for checkerboard calibration using SkyGuard's USB source."""
import argparse
from pathlib import Path
import sys
import tkinter as tk

import cv2
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.video.source import UniversalVideoSource, VideoSourceConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)

    with UniversalVideoSource(VideoSourceConfig("usb", args.index)) as camera:
        root = tk.Tk()
        root.title("SkyGuard USB checkerboard capture")
        preview = tk.Label(root)
        preview.pack()
        status = tk.Label(root, text="Space: save the original frame. Esc: close.")
        status.pack()
        latest = None
        index = 1

        def save(_event=None):
            nonlocal index
            if latest is None:
                return
            path = args.directory / f"{index:02d}-checkerboard.png"
            while path.exists():
                index += 1
                path = args.directory / f"{index:02d}-checkerboard.png"
            ok, encoded = cv2.imencode(".png", latest.image)
            if not ok:
                status.config(text="PNG encoding failed")
                return
            with path.open("xb") as stream:
                stream.write(encoded.tobytes())
            status.config(text=f"Saved {path.name} · Space: save next · Esc: close")
            index += 1

        def update():
            nonlocal latest
            frame = camera.read()
            if frame is None:
                latest = None
                status.config(text="USB camera is not returning a frame")
            elif (frame.width, frame.height) != (args.width, args.height):
                latest = None
                status.config(text=f"USB returned {frame.width} × {frame.height}; expected {args.width} × {args.height}")
            else:
                latest = frame
                image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)))
                preview.config(image=image)
                preview.image = image
            root.after(35, update)

        root.bind("<space>", save)
        root.bind("<Escape>", lambda _event: root.destroy())
        update()
        root.mainloop()


if __name__ == "__main__":
    main()
