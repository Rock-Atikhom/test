import os
import yaml
import time
import math
import cv2
import numpy as np
from collections import defaultdict, deque
from video_stream import VideoStream
from inference import YOLOInferenceEngine

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# --- Line Segment Intersection Helpers ---
def ccw(A, B, C):
    """Check if three points are listed in counter-clockwise order."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def intersect(A, B, C, D):
    """Return True if line segments AB and CD intersect."""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def get_crossing_direction(A, B, C, D):
    """Determine crossing direction based on line slope orientation."""
    dx_line = D[0] - C[0]
    dy_line = D[1] - C[1]
    
    if abs(dx_line) >= abs(dy_line):
        # Horizontal-ish line: compare Y coordinates (downward is incoming, upward is outgoing)
        return "Incoming" if B[1] > A[1] else "Outgoing"
    else:
        # Vertical-ish line: compare X coordinates (rightward is incoming, leftward is outgoing)
        return "Incoming" if B[0] > A[0] else "Outgoing"

# --- HSV dominant color classification ---
def detect_vehicle_color(crop_bgr):
    """Extract dominant vehicle color using HSV color ranges on center ROI."""
    if crop_bgr is None or crop_bgr.size == 0:
        return "Unknown"
    
    # Resize to speed up calculation and smooth out details
    crop_resized = cv2.resize(crop_bgr, (40, 40))
    hsv = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Bins definition: Black, White, Gray
    black_mask = (v < 45)
    white_mask = (s < 35) & (v > 175)
    gray_mask = (s < 35) & (v >= 45) & (v <= 175)
    
    # Filter colored pixels
    color_mask = ~(black_mask | white_mask | gray_mask)
    h_colored = h[color_mask]
    
    # Count matching pixels for each bin
    black_count = np.sum(black_mask)
    white_count = np.sum(white_mask)
    gray_count = np.sum(gray_mask)
    
    red_count = np.sum(((h_colored >= 0) & (h_colored <= 8)) | ((h_colored >= 170) & (h_colored <= 180)))
    orange_count = np.sum((h_colored >= 9) & (h_colored <= 20))
    yellow_count = np.sum((h_colored >= 21) & (h_colored <= 38))
    green_count = np.sum((h_colored >= 39) & (h_colored <= 85))
    blue_count = np.sum((h_colored >= 86) & (h_colored <= 130))
    purple_count = np.sum((h_colored >= 131) & (h_colored <= 169))
    
    color_scores = {
        "Black": black_count,
        "White": white_count,
        "Gray": gray_count,
        "Red": red_count,
        "Orange": orange_count,
        "Yellow": yellow_count,
        "Green": green_count,
        "Blue": blue_count,
        "Purple": purple_count
    }
    
    return max(color_scores, key=color_scores.get)

def center_of_box_xyxy(box_xyxy):
    x1, y1, x2, y2 = box_xyxy
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def process_and_draw(
    frame_bgr, 
    result, 
    track_history, 
    previous_centers, 
    previous_times, 
    class_names, 
    counting_cfg, 
    cumulative_counts, 
    crossed_ids,
    max_trail=30
):
    now = time.time()
    annotated = frame_bgr.copy()
    h_f, w_f = frame_bgr.shape[:2]

    # Get counting line coordinates from config
    line_coords = counting_cfg.get("line_coords", [0, h_f // 2, w_f, h_f // 2])
    line_p1 = (line_coords[0], line_coords[1])
    line_p2 = (line_coords[2], line_coords[3])

    if result.boxes is None or result.boxes.id is None:
        return annotated

    boxes = result.boxes.xyxy.cpu().numpy()
    track_ids = result.boxes.id.int().cpu().tolist()
    class_ids = result.boxes.cls.int().cpu().tolist()
    confidences = result.boxes.conf.cpu().numpy()

    for box, track_id, class_id, conf in zip(boxes, track_ids, class_ids, confidences):
        x1, y1, x2, y2 = box.astype(int)
        cx, cy = center_of_box_xyxy((x1, y1, x2, y2))

        # 1. Centroid History
        prev_cx, prev_cy = previous_centers.get(track_id, (cx, cy))
        track_history[track_id].append((cx, cy))
        if len(track_history[track_id]) > max_trail:
            track_history[track_id].popleft()

        # 2. Virtual Line Crossing Algorithm
        if counting_cfg.get("enabled", False) and track_id not in crossed_ids:
            if prev_cx != cx or prev_cy != cy: # Ensure motion occurred
                if intersect((prev_cx, prev_cy), (cx, cy), line_p1, line_p2):
                    crossed_ids.add(track_id)
                    direction = get_crossing_direction((prev_cx, prev_cy), (cx, cy), line_p1, line_p2)
                    class_name = class_names.get(class_id, "unknown")
                    # Increment specific class-based counts
                    if class_name in cumulative_counts:
                        cumulative_counts[class_name][direction] += 1
                        print(f"[Counting] ID {track_id} ({class_name}) crossed {direction}!")

        # 3. Vehicle Color Detection (HSV crop of center 60% of bbox)
        box_w, box_h = x2 - x1, y2 - y1
        crop_w, crop_h = int(box_w * 0.6), int(box_h * 0.6)
        crop_x1 = max(0, cx - crop_w // 2)
        crop_y1 = max(0, cy - crop_h // 2)
        crop_x2 = min(w_f, cx + crop_w // 2)
        crop_y2 = min(h_f, cy + crop_h // 2)
        
        crop = frame_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
        color_detected = detect_vehicle_color(crop)

        # 4. Speed & Direction Calculations
        speed_px_per_sec = 0.0
        movement_dir = ""
        if track_id in previous_centers:
            dt = max(now - previous_times.get(track_id, now), 1e-6)
            dx, dy = cx - prev_cx, cy - prev_cy
            speed_px_per_sec = math.sqrt(dx * dx + dy * dy) / dt
            if abs(dx) > abs(dy):
                movement_dir = "right" if dx > 0 else "left"
            else:
                movement_dir = "down" if dy > 0 else "up"

        previous_centers[track_id] = (cx, cy)
        previous_times[track_id] = now

        # Draw overlays
        color = (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)

        # Bounding box labels
        class_name = class_names.get(class_id, str(class_id))
        label = f"ID {track_id} {class_name} ({color_detected}) {conf:.2f}"
        motion = f"{movement_dir} {speed_px_per_sec:.0f}px/s" if movement_dir else "New"
        
        cv2.putText(annotated, label, (x1, max(20, y1 - 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, motion, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 2, cv2.LINE_AA)

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
    counting_cfg = cfg["counting"]
    
    print(f"--- CCTV Tracking, Counting & Color Detection ---")
    print(f"Execution Mode:   {execution_mode.upper()}")
    print(f"Stream Source:    {stream_cfg['source']}")
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
    
    # Setup tracking variables
    track_history = defaultdict(lambda: deque(maxlen=30))
    previous_centers = {}
    previous_times = {}
    crossed_ids = set()

    # Cumulative count table matching our COCO configurations
    # (car, motorcycle, bus, truck)
    cumulative_counts = {
        "car": {"Incoming": 0, "Outgoing": 0},
        "motorcycle": {"Incoming": 0, "Outgoing": 0},
        "bus": {"Incoming": 0, "Outgoing": 0},
        "truck": {"Incoming": 0, "Outgoing": 0}
    }
    
    # Frame rate calculation
    fps_start_time = time.time()
    frame_counter = 0
    calculated_fps = 0.0
    
    stream.start()
    
    try:
        while stream.started:
            frame = stream.read()
            if frame is None:
                time.sleep(0.01)
                continue
                
            frame_counter += 1
            h_f, w_f = frame.shape[:2]
            
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
            
            # 3. Draw Results and tracking trails
            annotated_frame = process_and_draw(
                frame,
                results[0],
                track_history,
                previous_centers,
                previous_times,
                class_names,
                counting_cfg,
                cumulative_counts,
                crossed_ids
            )
            
            # 4. Draw Counting Line Overlay
            if counting_cfg.get("enabled", False):
                line_coords = counting_cfg.get("line_coords", [0, h_f // 2, w_f, h_f // 2])
                # Yellow line to indicate active counting zone
                cv2.line(annotated_frame, (line_coords[0], line_coords[1]), (line_coords[2], line_coords[3]), (0, 255, 255), 2)
                cv2.putText(annotated_frame, "COUNTING LINE", (line_coords[0] + 10, line_coords[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

            # 5. Draw Cumulative Count Overlay Table
            # Draw semi-transparent background for counts table
            cv2.rectangle(annotated_frame, (10, 50), (250, 190), (0, 0, 0), -1)
            cv2.rectangle(annotated_frame, (10, 50), (250, 190), (255, 255, 255), 1)
            cv2.putText(annotated_frame, "Traffic Count Statistics", (15, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.line(annotated_frame, (10, 75), (250, 75), (255, 255, 255), 1)
            
            y_offset = 95
            for vehicle_class, dirs in cumulative_counts.items():
                stats = f"{vehicle_class.capitalize()}: In: {dirs['Incoming']} | Out: {dirs['Outgoing']}"
                cv2.putText(annotated_frame, stats, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                y_offset += 22

            # Rolling FPS
            now = time.time()
            dt = now - fps_start_time
            if dt >= 1.0:
                calculated_fps = frame_counter / dt
                frame_counter = 0
                fps_start_time = now
                
            fps_text = f"FPS: {calculated_fps:.1f} | Mode: {execution_mode.upper()}"
            cv2.putText(annotated_frame, fps_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
            
            # Display Window
            try:
                cv2.imshow("Jetson CV CCTV Tracking", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except cv2.error:
                if frame_counter == 1:
                    print("[Main] Warning: GUI window unavailable (headless). Tracking active in background...")
                
            time.sleep(0.001)
            
    except KeyboardInterrupt:
        print("[Main] Process interrupted.")
    finally:
        stream.stop()
        cv2.destroyAllWindows()
        print("[Main] Cleanup complete.")

if __name__ == "__main__":
    main()
