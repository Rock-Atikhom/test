import argparse
import sys
from ultralytics import YOLO

def export_model(model_path, imgsz=320, half=True):
    print(f"[Export] Loading model from: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"[Export] Error loading model: {e}")
        sys.exit(1)
        
    print(f"[Export] Compiling model to TensorRT engine format (imgsz={imgsz}, FP16={half})...")
    print("[Export] Note: This compilation process can take up to 10-15 minutes on the Jetson Nano. Please be patient.")
    
    try:
        # Export model to TensorRT format
        # device=0 refers to the GPU (Jetson Nano has a single integrated Maxwell GPU)
        engine_file = model.export(
            format="engine",
            imgsz=imgsz,
            half=half,
            device=0,
            dynamic=False
        )
        print(f"[Export] Success! Compiled TensorRT engine generated: {engine_file}")
    except Exception as e:
        print(f"[Export] Failed to compile TensorRT engine: {e}")
        print("[Export] Check if TensorRT is correctly installed in your JetPack environment.")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile PyTorch YOLO models to TensorRT engine format on Jetson Nano")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Path to PyTorch model (.pt file)")
    parser.add_argument("--imgsz", type=int, default=320, help="Image size resolution for model input (default: 320)")
    parser.add_argument("--no-half", action="store_true", help="Disable FP16 precision compilation (forces FP32)")
    
    args = parser.parse_args()
    
    export_model(
        model_path=args.model,
        imgsz=args.imgsz,
        half=not args.no_half
    )
