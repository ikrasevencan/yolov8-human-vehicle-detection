# yolov8-human-vehicle-detection
AI-based real-time human and vehicle detection system using YOLOv8 for video and camera streams.
AI-Based Human & Vehicle Detection System (YOLOv8)
A real-time computer vision system that detects humans and vehicles (car, truck, bus, motorcycle, bicycle) using YOLOv8.
The system supports image, video, and live webcam inputs and is optimized for practical testing on real-world scenes.
Project Overview
This project implements an AI-based object detection pipeline using Ultralytics YOLOv8 to identify and localize humans and vehicles in various environments.
During development, the model was tested on:
Street images containing pedestrians
Traffic scenes with multiple vehicles
Real-world video frames from urban environments
Webcam input
The goal is to simulate real-world surveillance and traffic monitoring scenarios.
Technologies Used
Python 3.x
YOLOv8
OpenCV
PyTorch
NumPy
Project Structure
yolov8_detector/
├── detector.py        # Core detection (image/video/webcam)
├── batch_detect.py    # Batch image processing
├── zone_counter.py    # Line / zone counting system
├── requirements.txt
└── output/            # Saved detection results
 Installation
# Clone the repository
cd yolov8_detector

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Testing Notes
The system was evaluated using:
Real street pedestrian images (people walking in urban environments)
Traffic videos with multiple vehicle types
Mixed real-world scenes captured from public environments
Webcam-based live detection
These tests helped verify robustness under different lighting and density conditions.
Detection Classes
Person
Car
Truck
Bus
Motorcycle
Bicycle
Each detection includes:
Bounding box coordinates
Class label
Confidence score
 Features
Real-time object detection
Multi-class recognition (humans + vehicles)
Video, image, and webcam support
Batch image processing
Zone-based counting system
Output saving with annotations
Future Improvements
Object tracking (ID-based tracking across frames)
Traffic density analysis
Automatic alert system for restricted zones
Edge deployment (Jetson Nano / Raspberry Pi)
Model optimization for higher FPS
Example Use Case
This system can be used for:
Traffic monitoring systems
Security surveillance
Smart city applications
AI-based environment analysis
 Author
Developed by: İkra Sevencan
