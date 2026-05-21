import os
import cv2
import numpy as np
import torch

class TrackedObject:
    def __init__(self, bbox, track_id, class_id, confidence):
        """Unified object container for tracking results across all model backends."""
        self.bbox = bbox          # [x1, y1, x2, y2] bounding box coordinates
        self.track_id = track_id  # Unique ID assigned by the tracker
        self.class_id = class_id  # Bounding box object category ID
        self.confidence = confidence # Bounding box prediction confidence

class YOLOInferenceEngine:
    def __init__(self, model_path="yolo11n.pt", mode="local"):
        self.model_path = model_path
        self.mode = mode
        self.model = None
        self.load_model()

    def load_model(self):
        from ultralytics import YOLO
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
                    print(f"[Inference] Falling back to PyTorch model: {self.model_path}")
                    self.model = YOLO(self.model_path)
                else:
                    raise FileNotFoundError(f"Neither {engine_path} nor {self.model_path} could be loaded.")
        else:
            # Local Mode: Load PyTorch model directly (CPU / MPS / CUDA)
            print(f"[Inference] Loading PyTorch model in local mode: {self.model_path}")
            self.model = YOLO(self.model_path)

    def track(self, frame, persist=True, tracker="bytetrack.yaml", imgsz=320, conf=0.35, classes=None):
        """Execute tracking on the frame and return standardized TrackedObjects."""
        if self.model is None:
            raise RuntimeError("Model is not loaded.")
        
        results = self.model.track(
            source=frame,
            persist=persist,
            tracker=tracker,
            imgsz=imgsz,
            conf=conf,
            classes=classes,
            verbose=False
        )
        
        tracked_objects = []
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            confs = results[0].boxes.conf.cpu().numpy()
            
            for box, tid, cid, c_score in zip(boxes, track_ids, class_ids, confs):
                tracked_objects.append(TrackedObject(box, tid, cid, float(c_score)))
                
        return tracked_objects


class RFDETRInferenceEngine:
    def __init__(self, model_path="rfdetr-nano", mode="local", conf=0.35):
        self.model_path = model_path
        self.mode = mode
        self.conf = conf
        self.model = None
        self.ort_session = None
        self.tracker = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()

    def load_model(self):
        import supervision as sv
        # Always use Supervision's ByteTrack for RF-DETR tracking
        self.tracker = sv.ByteTrack(track_activation_threshold=self.conf)
        
        # Determine if loading from local compiled ONNX/TensorRT or downloading HF PyTorch model
        if self.model_path.endswith((".onnx", ".engine", ".trt")):
            # Load using ONNX Runtime
            print(f"[Inference] Loading RF-DETR model via ONNX Runtime: {self.model_path}")
            try:
                import onnxruntime as ort
                providers = [
                    'TensorrtExecutionProvider', 
                    'CUDAExecutionProvider', 
                    'CPUExecutionProvider'
                ]
                self.ort_session = ort.InferenceSession(self.model_path, providers=providers)
                print(f"[Inference] ONNX Runtime loaded with providers: {self.ort_session.get_providers()}")
            except ImportError:
                print("[Inference] Error: onnxruntime is required to run .onnx RF-DETR models. Run `pip install onnxruntime`.")
                raise
        else:
            # Load using Roboflow RF-DETR Python library
            print(f"[Inference] Loading RF-DETR model ({self.model_path}) on device: {self.device}")
            from rfdetr import RFDETRNano
            
            # Instantiate model
            self.model = RFDETRNano(device=self.device)
            
            # Apply half precision if on CUDA (Jetson/GPU deployment)
            if self.device == "cuda":
                print("[Inference] Enabling FP16 half precision for RF-DETR on CUDA...")
                self.model.model = self.model.model.half()

    def track(self, frame, persist=True, tracker="bytetrack.yaml", imgsz=320, conf=0.35, classes=None):
        """Execute inference and tracking using RF-DETR."""
        import supervision as sv
        
        # 1. Image preprocessing & detection
        if self.ort_session is not None:
            # ONNX Runtime Inference Pipeline
            h, w = frame.shape[:2]
            # RF-DETR Nano requires RGB input and specific square size (e.g. 320x320)
            input_size = (imgsz, imgsz)
            resized = cv2.resize(frame, input_size)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0.0, 1.0] and transpose to BCHW format
            blob = rgb.astype(np.float32) / 255.0
            blob = np.transpose(blob, (2, 0, 1)) # HWC -> CHW
            blob = np.expand_dims(blob, axis=0) # CHW -> BCHW
            
            # Run inference session
            inputs = {self.ort_session.get_inputs()[0].name: blob}
            outputs = self.ort_session.run(None, inputs)
            
            # Parse predictions from ONNX output (expecting bounding boxes, scores, and labels)
            # Depending on model export, typically returns: boxes, scores, classes
            raw_boxes, raw_scores, raw_classes = outputs[0][0], outputs[1][0], outputs[2][0]
            
            # Convert boxes back to original resolution
            scale_x, scale_y = w / imgsz, h / imgsz
            boxes = []
            confidences = []
            class_ids = []
            
            for box, score, class_id in zip(raw_boxes, raw_scores, raw_classes):
                if score >= conf:
                    x1, y1, x2, y2 = box
                    boxes.append([x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y])
                    confidences.append(float(score))
                    class_ids.append(int(class_id))
            
            detections = sv.Detections(
                xyxy=np.array(boxes) if len(boxes) > 0 else np.empty((0, 4)),
                confidence=np.array(confidences) if len(confidences) > 0 else np.empty(0),
                class_id=np.array(class_ids) if len(class_ids) > 0 else np.empty(0)
            )
        else:
            # PyTorch Inference Pipeline
            # Convert OpenCV BGR to RGB channel order for RF-DETR
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Perform prediction
            detections = self.model.predict(frame_rgb, threshold=conf)
            
        # 2. Filter detections by classes if specified
        if classes is not None and len(classes) > 0 and len(detections) > 0:
            mask = np.isin(detections.class_id, classes)
            detections = detections[mask]
            
        # 3. Update ByteTrack
        tracked_detections = self.tracker.update_with_detections(detections)
        
        # 4. Convert to standardized TrackedObject list
        tracked_objects = []
        if tracked_detections.tracker_id is not None:
            for box, tid, cid, conf_score in zip(
                tracked_detections.xyxy, 
                tracked_detections.tracker_id, 
                tracked_detections.class_id, 
                tracked_detections.confidence
            ):
                tracked_objects.append(TrackedObject(box, int(tid), int(cid), float(conf_score)))
                
        return tracked_objects
