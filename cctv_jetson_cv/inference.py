import os
from ultralytics import YOLO

class YOLOInferenceEngine:
    def __init__(self, model_path="yolo11n.pt", mode="local"):
        self.model_path = model_path
        self.mode = mode
        self.model = None
        self.load_model()

    def load_model(self):
        if self.mode == "jetson":
            # On Jetson Nano, check if we have a TensorRT engine version of the model
            base, ext = os.path.splitext(self.model_path)
            engine_path = f"{base}.engine"
            
            if os.path.exists(engine_path):
                print(f"[Inference] Loading optimized TensorRT engine: {engine_path}")
                self.model = YOLO(engine_path, task="detect")
            else:
                print(f"[Inference] Warning: TensorRT engine not found at {engine_path}")
                if os.path.exists(self.model_path) and self.model_path.endswith(".pt"):
                    print(f"[Inference] Falling back to PyTorch model: {self.model_path} (Note: FPS will be low on Jetson CPU/GPU without TensorRT compile)")
                    self.model = YOLO(self.model_path)
                else:
                    raise FileNotFoundError(f"Neither {engine_path} nor {self.model_path} could be loaded.")
        else:
            # Local Mode: Load PyTorch model directly (CPU / MPS / CUDA)
            print(f"[Inference] Loading PyTorch model in local mode: {self.model_path}")
            self.model = YOLO(self.model_path)

    def track(self, frame, persist=True, tracker="bytetrack.yaml", imgsz=320, conf=0.35, classes=None):
        """Execute tracking on the given frame."""
        if self.model is None:
            raise RuntimeError("Model is not loaded.")
        
        # Run tracking using the appropriate backend
        results = self.model.track(
            source=frame,
            persist=persist,
            tracker=tracker,
            imgsz=imgsz,
            conf=conf,
            classes=classes,
            verbose=False
        )
        return results
