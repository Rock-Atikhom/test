import threading
import time
import cv2

class VideoStream:
    def __init__(self, source=0, width=640, height=480, fps=30, mode="local"):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.mode = mode
        
        # Connection parameters
        self.cap = None
        self.frame = None
        self.started = False
        self.read_lock = threading.Lock()
        self.thread = None

    def start(self):
        if self.started:
            return self
        
        # Build source pipeline or capture object
        if self.mode == "jetson" and isinstance(self.source, str) and self.source.startswith("rtsp://"):
            # Hardware-accelerated GStreamer decoding for Jetson Nano
            gstreamer_pipeline = (
                f"rtspsrc location={self.source} latency=200 ! "
                "rtph264depay ! h264parse ! nvv4l2decoder ! "
                f"nvvidconv ! video/x-raw, width={self.width}, height={self.height}, format=BGRx ! "
                "appsink drop=true sync=false"
            )
            print(f"[VideoStream] Starting Jetson GStreamer pipeline:\n{gstreamer_pipeline}")
            self.cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
        else:
            # Local / WebCam / Video File capture
            print(f"[VideoStream] Starting standard OpenCV capture on source: {self.source}")
            self.cap = cv2.VideoCapture(self.source)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not self.cap or not self.cap.isOpened():
            print(f"[VideoStream] Error: Could not open source: {self.source}")
            # Try fallback to standard capture if GStreamer fails
            if self.mode == "jetson":
                print("[VideoStream] GStreamer failed. Falling back to standard OpenCV RTSP capture...")
                self.cap = cv2.VideoCapture(self.source)
                if not self.cap.isOpened():
                    raise RuntimeError(f"Failed to open source {self.source} even with fallback.")
            else:
                raise RuntimeError(f"Failed to open source {self.source}")

        # Read first frame to initialize
        ok, self.frame = self.cap.read()
        if not ok:
            print("[VideoStream] Error: Failed to read initial frame.")

        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        delay = 1.0 / self.fps
        while self.started:
            ok, frame = self.cap.read()
            if not ok:
                print("[VideoStream] Stream ended or connection lost.")
                self.started = False
                break
                
            # If format BGRx (from GStreamer), convert to BGR
            if frame is not None and frame.shape[-1] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            with self.read_lock:
                self.frame = frame
                
            time.sleep(delay)

    def read(self):
        with self.read_lock:
            frame_copy = self.frame.copy() if self.frame is not None else None
        return frame_copy

    def stop(self):
        self.started = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        print("[VideoStream] Stream stopped.")
