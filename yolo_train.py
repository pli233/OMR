from ultralytics import YOLO
import torch

# 1. Load a pretrained YOLOv8 medium (yolov8m) model
model = YOLO('yolov8m.pt') 

# 2. Check if a CUDA-compatible GPU is available
device = torch.device("cuda:0")

# 3. If GPU is available, configure training to use GPU
if torch.cuda.is_available():
    print('Training with GPU.')
    results = model.train(
        data="deepscores.yaml",    # 3.1 Path to dataset config file
        epochs= 300,               # 3.2 Number of training epochs, usually 300 epochs has made the best result
        batch=3,                   # 3.3 Batch size per iteration
        device=device,             # 3.4 Use GPU device
        patience=0,                # 3.5 No early stopping
        deterministic=True,        # 3.6 Ensure reproducible results
        amp=True                   # 3.7 Enable automatic mixed precision for speed
    )

# 4. If GPU is not available, fall back to CPU training 
else:
    print('Training with CPU, it will be really slow')
    results = model.train(data="deepscores.yaml", imgsz=640, epochs=10, batch=3)
