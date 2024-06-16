from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('../weights/pose_estimation/yolov8l-pose.pt')

# Export the model to TensorRT format
model.export(format='engine')

# Load the exported TensorRT model
tensorrt_model = YOLO('../weights/pose_estimation/yolov8l-pose.engine')