from ultralytics import YOLO

model = YOLO("yolov8m.pt")

results = model.train(
    data="dataset/data.yaml",
    epochs=75,
    imgsz=640,
    device="cpu"
)