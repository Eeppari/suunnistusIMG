"""
Convert TANVOM's raw masks into YOLO-segmentation polygon labels,
writing directly into a `labels/` folder next to each `images/` folder
(Ultralytics finds labels automatically via the images->labels path swap).
 
Expected input layout (as shipped by the dataset):
 
    <TANVOM_ROOT>/processed/tiles_512_s128_v1/
        train/images/*.png
        train/masks/contours/101_102_Contour/*.png
        train/masks/nav/503_508_RoadsBlack/*.png
        train/masks/terrain_rare/107/*.png
        train/masks/terrain_rare/108/*.png
        train/masks/region_overlay/407/*.png
        train/masks/region_overlay/409/*.png
        val/...  (same layout)
 
Output:
    train/labels/*.txt, val/labels/*.txt   (one file per image, YOLO-seg polygons)
 
Usage:
    pip install opencv-python numpy
    python convert_masks_to_yolo.py
"""
 
from pathlib import Path
from collections import defaultdict
 
import cv2
import numpy as np
 
# ---- Settings: edit these ----
TANVOM_ROOT = Path("TANVOM_dataset")  # folder where you extracted the dataset
TILES_DIR = TANVOM_ROOT / "processed" / "tiles_512_s128_v1"
SPLITS = ["train", "val"]
 
# class_id -> mask subfolder (relative to <split>/masks/)
# This order also defines the class ids used in data.yaml — keep them in sync.
CLASS_SOURCES = {
    0: "contours/101_102_Contour",
    1: "nav/503_508_RoadsBlack",
    2: "terrain_rare/107",
    3: "terrain_rare/108",
    4: "region_overlay/407",
    5: "region_overlay/409",
}
# region_landcover is a multi-class pixel-id map (different task) — not included here.
 
MIN_CONTOUR_AREA = 15         # skip tiny noise blobs (pixels)
EPSILON_FACTOR = 0.002        # polygon simplification (relative to contour perimeter)
# --------------------------------
 
 
def mask_to_polygons(mask: np.ndarray, class_id: int):
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
    lines = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
            continue
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, EPSILON_FACTOR * perimeter, True)
        if len(approx) < 3:
            continue
        coords = []
        for point in approx.reshape(-1, 2):
            x, y = point
            coords.append(f"{x / w:.6f}")
            coords.append(f"{y / h:.6f}")
        lines.append(f"{class_id} " + " ".join(coords))
    return lines
 
 
def process_split(split: str):
    images_dir = TILES_DIR / split / "images"
    masks_dir = TILES_DIR / split / "masks"
    labels_dir = TILES_DIR / split / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
 
    # stem -> list of label lines, accumulated across all classes
    lines_by_stem = defaultdict(list)
 
    for class_id, subfolder in CLASS_SOURCES.items():
        class_dir = masks_dir / subfolder
        if not class_dir.exists():
            print(f"  ! missing folder, skipping: {class_dir}")
            continue
 
        mask_files = sorted(class_dir.glob("*.png")) + sorted(class_dir.glob("*.jpg"))
        total = len(mask_files)
        print(f"  [{split}] {subfolder}: {total} mask files")
 
        for i, mask_path in enumerate(mask_files, start=1):
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                continue
            binary = (mask_img > 0).astype(np.uint8) * 255
            lines = mask_to_polygons(binary, class_id)
            if lines:
                lines_by_stem[mask_path.stem].extend(lines)
 
            if i % 500 == 0 or i == total:
                print(f"    {i}/{total}", end="\r", flush=True)
        print()  # move to a new line after the last \r update
 
    # write one label file per image (empty file if no objects -> background tile)
    image_files = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
    total_images = len(image_files)
    print(f"  [{split}] writing labels for {total_images} images...")
 
    n_with_objects = 0
    for i, img_path in enumerate(image_files, start=1):
        out_path = labels_dir / (img_path.stem + ".txt")
        lines = lines_by_stem.get(img_path.stem, [])
        if lines:
            n_with_objects += 1
        out_path.write_text("\n".join(lines))
 
        if i % 1000 == 0 or i == total_images:
            print(f"    {i}/{total_images}", end="\r", flush=True)
    print()
 
    print(f"[{split}] done — {total_images} images, {n_with_objects} with at least one object")
 
 
def main():
    for split in SPLITS:
        process_split(split)
 
 
if __name__ == "__main__":
    main()