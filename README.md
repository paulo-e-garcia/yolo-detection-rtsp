# YOLO Object Detection RTSP Stream

Real-time object detection on RTSP video streams using YOLOv9 with web-based visualization.

## Overview

This script processes an RTSP stream (from a camera or other sources) through a YOLOv9 model to detect specific objects (people, vehicles, pets) and streams the annotated video to a web browser via MJPEG over HTTP.

## How It Works
	1.	Connects to RTSP stream with authentication
	2.	Processes frames through YOLOv9 model on GPU
	3.	Filters detections by configured classes
	4.	Draws bounding boxes and confidence scores
	5.	Encodes frames as MJPEG
	6.	Serves video stream via HTTP on port 8080

## Features

- Real-time object detection using YOLOv9e model
- GPU acceleration (CUDA/ROCm/CPU support)
- RTSP stream input with authentication
- Web-based output via MJPEG streaming
- Configurable detection classes
- Multithreaded frame processing

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- Ultralytics YOLO
- CUDA-capable GPU (recommended) or AMD GPU with ROCm

## Configuration

### Detection Classes

Modify the `DETECT_CLASSES` list to specify which objects to detect, according to the capabilities of the YOLO model version you'll use:

```python
DETECT_CLASSES = ['person', 'cat', 'dog', 'motorcycle', 'bicycle', 'truck', 'car']
```

## Model Settings
	•	Model: `yolov9e.pt` # You can change this as you wish
	•	Device: `cuda` (change to `hip` for AMD/ROCm or `cpu` for CPU-only)
	•	Confidence threshold: 0.7 # Change it as you need
	•	Half precision: Enabled for faster inference

## RTSP Stream

Update the RTSP connection parameters:

```python
rtsp_user = '<usuario>'
rtsp_pass = '<senha>'
rtsp_url = f'rtsp://{rtsp_user}:{rtsp_pass}@<IP>:<Porta><URL>'
```
Note: if your RTSP stream doesn't need username and password, edit the rtsp_url variable below and remove the parameters from the URL.

## HTTP Server

Configure the output server address:

```python
HTTPServer(('<IP>', 8080), MJPEGHandler).serve_forever()
```
## Usage

Run the script:

```python
python yolo-detection-rtsp.py
```
Access the detection stream in your browser: http://<IP>:8080/stream.mjpg

## Alternative Display Method

If running locally, or your remote server has X11 remote display configured, replace the HTTP server section with:

```python
cv2.imshow('Detection', frame)
if cv2.waitKey(1) & 0xFF == ord('q'):
    break
```