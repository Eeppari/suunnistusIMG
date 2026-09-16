from ultralytics import YOLO
 
model = YOLO("yolo11n-seg.pt")   # segmentation variant (note the "-seg")
model.train(
    data="data.yaml",
    epochs=10,        # quick sanity-check run; raise once this works
    imgsz=320,
    batch=5,         # bigger batch = fewer iterations/epoch = less CPU overhead
    fraction=0.4,     # train on 10% of the data (~1888 images) for a fast check
    workers=4,        # use a few CPU cores for data loading instead of 0
)