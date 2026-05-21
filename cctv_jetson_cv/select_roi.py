import os
import re
import cv2
import yaml
import numpy as np

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def write_config_preserve_comments(config_path, pts):
    with open(config_path, "r") as f:
        content = f.read()
    
    lines = content.splitlines()
    in_roi = False
    new_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        # Check if we enter the roi block
        if stripped.startswith("roi:"):
            in_roi = True
            new_lines.append(line)
            continue
            
        if in_roi:
            # If we hit another unindented key, we are out of the roi block
            if line.strip() and not line.startswith(" ") and not line.startswith("#"):
                in_roi = False
                new_lines.append(line)
                continue
                
            if stripped.startswith("enabled:"):
                indent = line[:line.find("enabled:")]
                new_lines.append(f"{indent}enabled: true")
            elif stripped.startswith("polygon:"):
                indent = line[:line.find("polygon:")]
                new_lines.append(f"{indent}polygon: {pts}")
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
            
    with open(config_path, "w") as f:
        f.write("\n".join(new_lines) + "\n")

# Global variables for mouse callback
pts = []
window_name = "Select ROI - Click to add points. Press 's' to Save, 'c' to Clear, 'q' to Quit"

def mouse_callback(event, x, y, flags, param):
    global pts
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append([x, y])
        print(f"Added point: [{x}, {y}]")

def main():
    global pts
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if not os.path.exists(config_path):
        print(f"Error: config.yaml not found at {config_path}")
        return

    cfg = load_config(config_path)
    source = cfg["stream"]["source"]
    width = cfg["stream"].get("width", 1280)
    height = cfg["stream"].get("height", 720)

    print(f"Connecting to stream: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open source: {source}")
        return

    # Try to set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Read a few frames to let the sensor warm up
    frame = None
    for _ in range(15):
        ret, temp_frame = cap.read()
        if ret:
            frame = temp_frame

    cap.release()

    if frame is None:
        print("Error: Could not grab frame from stream.")
        return

    # Create window and register mouse callback
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\nControls:")
    print("  Left Click: Add a vertex to the ROI polygon")
    print("  's' or Enter: Save the polygon to config.yaml and enable ROI")
    print("  'c': Clear the current selection")
    print("  'q' or Esc: Quit without saving")

    while True:
        display_frame = frame.copy()
        
        # Draw current points and lines
        if len(pts) > 0:
            for i in range(len(pts)):
                cv2.circle(display_frame, tuple(pts[i]), 5, (0, 0, 255), -1)
                cv2.putText(display_frame, str(i), (pts[i][0] + 8, pts[i][1] - 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                if i > 0:
                    cv2.line(display_frame, tuple(pts[i-1]), tuple(pts[i]), (0, 255, 0), 2)
            
            # Close the polygon overlay if there are >= 3 points
            if len(pts) >= 3:
                cv2.line(display_frame, tuple(pts[-1]), tuple(pts[0]), (0, 255, 0), 2)
                pts_array = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                overlay = display_frame.copy()
                cv2.fillPoly(overlay, [pts_array], (255, 0, 0)) # blue fill
                cv2.addWeighted(overlay, 0.15, display_frame, 0.85, 0, display_frame)

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('s') or key == 13: # 's' or Enter
            if len(pts) < 3:
                print("Warning: An ROI polygon needs at least 3 points. Click more points before saving!")
                continue
            write_config_preserve_comments(config_path, pts)
            print(f"Successfully saved {len(pts)} points to config.yaml and enabled ROI.")
            break
        elif key == ord('c'):
            pts = []
            print("Cleared points.")
        elif key == ord('q') or key == 27: # 'q' or Esc
            print("Quit without saving.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
