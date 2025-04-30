from ultralytics import YOLO
import torch

model = YOLO('yolov8m.pt') 
# Check if CUDA is available and configure training accordingly

device = torch.device("cuda:0")

if torch.cuda.is_available():
    print('Training with GPU.')
    results = model.train(data="deep_scores.yaml", epochs=500, batch=4, imgsz=[704, 992], device=device, patience=0, deterministic=True, amp=True) # rect=True, ,[928, 1312], imgsz=[704, 992], imgsz=[1960, 2772]
    
else:
    results = model.train(data="deep_scores.yaml", imgsz=640, rect=True, epochs = 10, batch=4)  # patience=100, device="mps", deterministic=True, amp=True