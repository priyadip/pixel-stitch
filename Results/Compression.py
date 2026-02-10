from PIL import Image
import os

Image.MAX_IMAGE_PIXELS = None   # allow very large images
MAX_SIZE = 5 * 1024 * 1024      # 5 MB limit


def process_png(path):
    if os.path.getsize(path) <= MAX_SIZE:
        print("Skipped (<5MB):", path)
        return

    img = Image.open(path).convert("RGB")

    # replace .png with .jpg (same name, same folder)
    new_path = os.path.splitext(path)[0] + ".jpg"

    img.save(new_path, "JPEG", quality=85, optimize=True)

    print("Converted to JPG:", new_path)


base_dir = "."

for folder in os.listdir(base_dir):
    if folder.startswith("AppendixF"):
        appendix_path = os.path.join(base_dir, folder)

        # 1️⃣ Images directly inside Appendix folder
        for file in os.listdir(appendix_path):
            path = os.path.join(appendix_path, file)
            if file.lower().endswith(".png") and os.path.isfile(path):
                process_png(path)

        # 2️⃣ Images inside output folder
        output_path = os.path.join(appendix_path, "output")
        if os.path.isdir(output_path):
            for file in os.listdir(output_path):
                path = os.path.join(output_path, file)
                if file.lower().endswith(".png"):
                    process_png(path)

print("Done processing all appendix images.")
