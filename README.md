# TANVOM → YOLO Segmentation

Only 3 files — everything else lives inside your extracted TANVOM dataset folder.

```
tanvom_yolo/
├── convert_masks_to_yolo.py   # masks -> YOLO polygon labels (writes into TANVOM's own folders)
├── data.yaml
└── train.py
```

## 1. Extract the dataset
Put the extracted TANVOM dataset next to these files (or edit `TANVOM_ROOT` in
`convert_masks_to_yolo.py` / the paths in `data.yaml` to point wherever it lives):

```
tanvom_yolo/
└── TANVOM_dataset/
    └── processed/
        ├── road_classifier/                  # different task, not used here
        └── tiles_512_s128_v1/
            ├── train/images, train/masks/...
            └── val/images,   val/masks/...
```

## 2. Which folders feed which class

| Dataset folder | Class |
|---|---|
| `masks/contours/101_102_Contour` | `contour` |
| `masks/nav/503_508_RoadsBlack` | `road_black` |
| `masks/terrain_rare/107` | `rare_107` |
| `masks/terrain_rare/108` | `rare_108` |
| `masks/region_overlay/407` | `overlay_407` |
| `masks/region_overlay/409` | `overlay_409` |

Not included (add later if you need them):
- `masks/region_landcover/landcover_id` — multi-class pixel-id map, different task from the
  binary masks above; needs its own handling since one file can carry many classes.
- `masks/region_landcover/landcover_rgb` — just a colorized visualization of the above.
- `previews/` — preview images, not training data.
- `road_classifier/black_roads_503_508_patches_v1/` — a separate patch-classification dataset
  for a later road-typing stage, not the tile segmentation task.
- val's `104_Earth_bank`, `107_Erosion_gully`, `108_Small_erosion_gully` folders — these look
  like differently-named duplicates of `107`/`108`; the script only reads the numeric folders
  to avoid double-counting objects. Check them yourself if val results look sparse for those
  classes — TANVOM may only ship the differently-named version there.

## 3. Convert masks to labels
```bash
pip install opencv-python numpy
python convert_masks_to_yolo.py
```
This writes `labels/` next to `images/` inside both `train/` and `val/` — e.g.
`tiles_512_s128_v1/train/labels/xxx.txt`. Ultralytics finds labels automatically by swapping
`images` → `labels` in the image path, so no extra config is needed.

## 4. Train
```bash
pip install ultralytics
python train.py
```
`train.py` uses `yolo11n-seg.pt` (the segmentation variant). `data.yaml` already points at
`TANVOM_dataset/processed/tiles_512_s128_v1/{train,val}/images` — edit the path there if you
extracted the dataset somewhere else.

## Notes
- The train/val split is already done at map-sheet level by the dataset itself — just use the
  `train/` and `val/` folders as they are, don't reshuffle.
- Tiles with no objects in the classes above still get an (empty) label file, which is correct
  — YOLO treats them as background examples.
- If you only care about one class at first (e.g. contours), trim `CLASS_SOURCES` in the
  script down to just that entry and update `nc`/`names` in `data.yaml` to match.
