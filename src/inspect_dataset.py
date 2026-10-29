from pathlib import Path
import cv2

DATA_DIR = Path("data/egg.v1i.folder")

extensions = {".jpg", ".jpeg", ".png", ".bmp"}

images = [
    p for p in DATA_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
]

print("Total images found:", len(images))

if not images:
    print("No images found. Check the dataset path.")
else:
    image_path = images[0]
    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not load:", image_path)
    else:
        height, width, channels = image.shape
        print("First image:", image_path)
        print("Width:", width)
        print("Height:", height)
        print("Channels:", channels)

        output_path = Path("outputs/first_image.jpg")
        cv2.imwrite(str(output_path), image)
        print("Preview saved to:", output_path)
