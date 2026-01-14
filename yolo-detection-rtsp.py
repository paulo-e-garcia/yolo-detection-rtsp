import cv2
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import time
from ultralytics import YOLO
from urllib.parse import quote

DETECT_CLASSES = ['person', 'cat', 'dog','motorcycle', 'bicycle','truck','car']

# Loads model to the GPU
model = YOLO('yolov9e.pt')
model.to('cuda') # Use "hip" in case of AMD (ROCm) or "cpu".

# RTSP credentials. Edit the rtsp_url below if your stream doesn't need any.
rtsp_user = '<usuario>'
rtsp_pass = encoded_password = quote('<senha>', safe='')

rtsp_url = f'rtsp://{rtsp_user}:{rtsp_pass}@<IP>:<Porta><URL>'
cap = cv2.VideoCapture(rtsp_url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

current_frame = None
frame_lock = threading.Lock()

def process_frames():
    global current_frame
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Run the detection model
        results = model(frame, verbose=False, conf=0.7, device='cuda', half=True)
        
        # Create bounding boxes with the confidence information
        for result in results:
            boxes = result.boxes.data.cpu().tolist()
            for box in boxes:
                x1, y1, x2, y2, confidence, class_id = box
                class_name = model.names[int(class_id)]
                
                if class_name in DETECT_CLASSES: # Filters detections based on the categories specified above. 
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(frame, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        with frame_lock:
            current_frame = frame.copy()

# The code block below creates an HTTP server to show the detection output on a web page. 
# This is not necessary if you're not running this on a remote server like me, or if you are able to 
# configure the X11 remote display properly (I couldn't). Replace it for cv2.imshow().
class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            
            while True:
                with frame_lock:
                    if current_frame is not None:
                        frame = current_frame.copy()
                    else:
                        continue
                
                ret, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')
                time.sleep(0.03)
        else:
            self.send_response(404)

threading.Thread(target=process_frames, daemon=True).start()

print("Output://<IP>:8080/stream.mjpg")
HTTPServer(('<IP>', 8080), MJPEGHandler).serve_forever()
