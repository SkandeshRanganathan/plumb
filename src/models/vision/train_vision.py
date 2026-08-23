import os
import argparse
from pathlib import Path
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

def train_pitch_detection(dataset_path: str, epochs: int = 50):
    """
    Trains a YOLOv8 model on the Pitch Detection dataset (satishchandala/cricket-pitch-detection-dataset).
    """
    if not YOLO:
        print("Ultralytics not installed. Run: pip install ultralytics")
        return
        
    print(f"Starting YOLOv8 Pitch Detection Training for {epochs} epochs...")
    model = YOLO("yolov8n.pt") # load a pretrained model
    
    yaml_path = os.path.join(dataset_path, "data.yaml")
    if not os.path.exists(yaml_path):
        print(f"data.yaml not found in {dataset_path}")
        return
        
    model.train(data=yaml_path, epochs=epochs, imgsz=640, project="models/saved/vision", name="pitch_detect")
    print("Pitch Detection Training Complete!")

def train_ball_tracking(dataset_path: str, epochs: int = 50):
    """
    Trains a YOLOv8 model on the Cricket Ball Dataset (kushagra3204/cricket-ball-dataset-for-yolo).
    """
    if not YOLO:
        print("Ultralytics not installed.")
        return
        
    print(f"Starting YOLOv8 Ball Tracking Training for {epochs} epochs...")
    model = YOLO("yolov8n.pt")
    
    yaml_path = os.path.join(dataset_path, "data.yaml")
    if not os.path.exists(yaml_path):
        print(f"data.yaml not found in {dataset_path}")
        return
        
    model.train(data=yaml_path, epochs=epochs, imgsz=640, project="models/saved/vision", name="ball_track")
    print("Ball Tracking Training Complete!")

def semantic_segmentation_info():
    """
    Instructions for the Semantic Segmentation dataset (sadhliroomyprime/cricket-semantic-segmentation)
    """
    print("Semantic Segmentation dataset requires PyTorch DeepLabV3 or U-Net architecture.")
    print("This is a massive 514MB dataset for Pixel-level classification of grass, dirt, and pitch cracks.")
    print("For live-stream speed, we currently use OpenCV color-histogram heuristics as a proxy.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["pitch", "ball", "segment"], required=True)
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    
    if args.task == "pitch":
        train_pitch_detection(args.dataset_path, args.epochs)
    elif args.task == "ball":
        train_ball_tracking(args.dataset_path, args.epochs)
    elif args.task == "segment":
        semantic_segmentation_info()
