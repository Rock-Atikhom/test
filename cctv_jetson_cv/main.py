import os
import yaml
import time
import math
import cv2
from collections import defaultdict, deque
from video_stream import VideoStream
from inference import YOLOInferenceEngine

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def enhance_bgr_with_lab_clahe(frame_bgr, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Enhance luminance in Lab space while keeping color channels stable."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

def center_of_box_xyxy(box_xyxy):
    x1, y1, x2, y2 = box_xyxy
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def draw_tracking_result(frame_bgr, result, track_history, previous_centers, previous_times, class_names, max_trail=30):
    now = time.time()
    annotated = frame_bgr.copy()

    if result.boxes is None or result.boxes.id is None:
        return annotated

    boxes = result.boxes.xyxy.cpu().numpy()
    track_ids = result.boxes.id.int().cpu().tolist()
    class_ids = result.boxes.cls.int().cpu().tolist()
    confidences = result.boxes.conf.cpu().numpy()

    for box, track_id, class_id, conf in zip(boxes, track_ids, class_ids, confidences):
        x1, y1, x2, y2 = box.astype(int)
        cx, cy = center_of_box_xyxy((x1, y1, x2, y2))

        # Store track coordinates
        track_history[track_id].append((cx, cy))
        if len(track_history[track_id]) > max_trail:
            track_history[track_id].popleft()

        # Speed and direction calculation
        speed_px_per_sec = 0.0
        direction = ""
        if track_id in previous_centers:
            prev_cx, prev_cy = previous_centers[track_id]
            dt = max(now - previous_times.get(track_id, now), 1e-6)
            dx, dy = cx - prev_cx, cy - prev_cy
            speed_px_per_sec = math.sqrt(dx * dx + dy * dy) / dt
            
            if abs(dx) > abs(dy):
                direction = "right" if dx > 0 else "left"
            else:
                direction = "down" if dy > 0 else "up"

        previous_centers[track_id] = (cx, cy)
        previous_times[track_id] = now

        # Draw bounding boxes and centroids
        color = (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)

        # Labels (IDs, Class names, Confidence)
        class_name = class_names.get(class_id, str(class_id))
        label = f"ID {track_id} {class_name} {conf:.2f}"
        motion = f"{direction} {speed_px_per_sec:.0f}px/s" if direction else "New Track"
        
        cv2.putText(annotated, label, (x1, max(20, y1 - 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, motion, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 0), 2, cv2.LINE_AA)

        # Draw trails
        trail = list(track_history[track_id])
        for i in range(1, len(trail)):
            cv2.line(annotated, trail[i - 1], trail[i], (0, 255, 255), 2)

    return annotated

def main():
    config_file = os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg = load_config(config_file)
    
    execution_mode = cfg["execution"]["mode"]
    stream_cfg = cfg["stream"]
    model_cfg = cfg["model"]
    pre_cfg = cfg["preprocessing"]
    
    print(f"--- CCTV Stream Tracking System ---")
    print(f"Execution Mode:   {execution_mode.upper()}")
    print(f"Stream Source:    {stream_cfg['source']}")
    print(f"Model Resolution: {model_cfg['imgsz']}x{model_cfg['imgsz']}")
    print(f"Enhance Lighting: {pre_cfg['use_lab_clahe']}")
    
    # Initialize Multi-threaded Video Stream
    stream = VideoStream(
        source=stream_cfg["source"],
        width=stream_cfg["width"],
        height=stream_cfg["height"],
        fps=stream_cfg["fps"],
        mode=execution_mode
    )
    
    # Initialize Inference Engine
    engine = YOLOInferenceEngine(
        model_path=model_cfg["model_path"],
        mode=execution_mode
    )
    
    class_names = engine.model.names if engine.model else {}
    
    # Tracking variables
    track_history = defaultdict(lambda: deque(maxlen=30))
    previous_centers = {}
    previous_times = {}
    
    # Frame rate calculation
    fps_start_time = time.time()
    frame_counter = 0
    calculated_fps = 0.0
    
    stream.start()
    
    try:
        while stream.started:
            frame = stream.read()
            if frame is None:
                # Wait for frames to populate
                time.sleep(0.01)
                continue
                
            frame_counter += 1
            
            # 1. Apply Optional Contrast Enhancement
            if pre_cfg["use_lab_clahe"]:
                processed_frame = enhance_bgr_with_lab_clahe(frame)
            else:
                processed_frame = frame.copy()
                
            # 2. Run Inference & Tracking
            results = engine.track(
                processed_frame,
                persist=True,
                tracker=model_cfg["tracker_config"],
                imgsz=model_cfg["imgsz"],
                conf=model_cfg["conf"],
                classes=model_cfg["classes"]
            )
            
            # 3. Draw Results
            annotated_frame = draw_tracking_result(
                frame,
                results[0],
                track_history,
                previous_centers,
                previous_times,
                class_names
            )
            
            # Calculate and overlay current FPS (rolling average)
            now = time.time()
            dt = now - fps_start_time
            if dt >= 1.0: # Update every second
                calculated_fps = frame_counter / dt
                frame_counter = 0
                fps_start_time = now
                
            fps_text = f"FPS: {calculated_fps:.1f} | Mode: {execution_mode.upper()}"
            cv2.putText(annotated_frame, fps_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
            
            # 4. Display Window
            try:
                cv2.imshow("Jetson CV CCTV Tracking", annotated_frame)
                # Exit on pressing 'q'
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except cv2.error:
                # Headless execution fallback
                if frame_counter == 1:
                    print("[Main] Warning: Display GUI is unavailable. Running in headless mode.")
                
            # Yield CPU resources briefly
            time.sleep(0.001)
            
    except KeyboardInterrupt:
        print("[Main] Process interrupted by user.")
    finally:
        stream.stop()
        cv2.destroyAllWindows()
        print("[Main] Cleanup complete.")

if __name__ == "__main__":
    main()
