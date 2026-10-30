from ultralytics import YOLO
if __name__ ==  '__main__':
    model = YOLO('yolov8n.pt')
    results = model.train(
        data = 'data.yaml',
        epochs = 116,
        imgsz = 640,
        batch = 16,
        workers = 0,
        name = 'cheremsha_clicker',
        patience = 56,
        copy_paste = 0.4,
        cls = 0.75
    )