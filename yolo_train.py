from ultralytics import YOLO
import torch
model = YOLO('yolov8m.pt') 

# Check if CUDA is available and configure training accordingly

device = torch.device("cuda:0")

if torch.cuda.is_available():
    print('Training with GPU.')
    results = model.train(data="deepscores.yaml", epochs=10, batch=3, device=device, patience=0, deterministic=True, amp=True)
    
else:
    results = model.train(data="deepscores.yaml", imgsz=640, rect=True, epochs = 10, batch=4)
