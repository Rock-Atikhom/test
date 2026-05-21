import argparse
import sys
import os

def export_model(model_path, imgsz=320, half=True):
    print(f"[Export] Loading model: {model_path}")
    
    # 1. RF-DETR Nano Export Handler
    if "rfdetr" in model_path.lower():
        try:
            from rfdetr import RFDETRNano
            print(f"[Export] Instantiating RFDETRNano for export...")
            model = RFDETRNano()
            
            # Export to ONNX format
            # rfdetr's export method outputs 'inference_model.onnx' in the specified directory
            output_dir = "."
            print(f"[Export] Exporting RF-DETR model to ONNX format (imgsz={imgsz})...")
            model.export(output_dir=output_dir)
            
            # Rename for clarity if necessary
            default_onnx = "inference_model.onnx"
            target_onnx = "rfdetr_nano.onnx"
            if os.path.exists(default_onnx):
                os.rename(default_onnx, target_onnx)
                print(f"[Export] Success! Exported ONNX model to: {os.path.abspath(target_onnx)}")
            else:
                print(f"[Export] Warning: Could not find output ONNX file at default path {default_onnx}.")
            
            # Print next-step trtexec instructions
            print("\n" + "="*70)
            print("NEXT STEP FOR JETSON NANO DEPLOYMENT:")
            print("Compile the exported ONNX model to a TensorRT engine using trtexec:")
            print(f"trtexec --onnx={target_onnx} --saveEngine=rfdetr_nano.engine --fp16 --memPoolSize=workspace:2048")
            print("="*70 + "\n")
            
        except Exception as e:
            print(f"[Export] Error exporting RF-DETR model: {e}")
            sys.exit(1)
            
    # 2. YOLO Export Handler
    else:
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
        except Exception as e:
            print(f"[Export] Error loading YOLO model: {e}")
            sys.exit(1)
            
        print(f"[Export] Compiling YOLO model to TensorRT engine format (imgsz={imgsz}, FP16={half})...")
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
    parser = argparse.ArgumentParser(description="Compile PyTorch models to TensorRT/ONNX format on Jetson Nano")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Path to PyTorch model (e.g. 'yolo11n.pt' or 'rfdetr-nano')")
    parser.add_argument("--imgsz", type=int, default=320, help="Image size resolution for model input (default: 320)")
    parser.add_argument("--no-half", action="store_true", help="Disable FP16 precision compilation for YOLO (forces FP32)")
    
    args = parser.parse_args()
    
    export_model(
        model_path=args.model,
        imgsz=args.imgsz,
        half=not args.no_half
    )
