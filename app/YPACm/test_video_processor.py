import json
import os
import tempfile
import time

import cv2
import numpy as np
import supervision as sv
import torch
from supervision import ColorAnnotator
from tqdm import tqdm
from ultralytics import YOLO

from app.YPACm.train_classifier.action_behavior_classifier.classifier import MultiPersonClassifier
from app.YPACm.train_classifier.utils.config import Config
from app.YPACm.utils.pose_annotator import HumanPoseAnnotator

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class VideoProcessor:
    def __init__(self):
        self.times = [0, 0, 0]
        self.frame_count = 0
        cfg = Config("YPACm/configs/infer_yolov8pose_dnn.yaml")
        get_path = lambda x: os.path.join(*x) if isinstance(x, (list, tuple)) else x
        pose_kwargs = cfg.POSE
        clf_kwargs = cfg.MULTI_CLASSIFIER
        self.model = YOLO(get_path(pose_kwargs['model_path']))
        self.tracker = sv.ByteTrack()
        self.multi_classifier = MultiPersonClassifier(**clf_kwargs)
        self.pose_annotator = HumanPoseAnnotator(thickness_skeleton=1)
        self.label_annotator = sv.LabelAnnotator()

    def callback(self, frame: np.ndarray, _: int) -> np.ndarray:
        start_time = time.time()
        results = self.model(frame, verbose=False, device=device, conf=0.5)[0]
        self.times[0] += time.time() - start_time

        start_time = time.time()
        detections = sv.Detections.from_ultralytics(results)
        # print(detections)
        detections = self.tracker.update_with_detections(detections)
        detections.data = {"keypoints": results.keypoints, "actions": [], "behaviors": []}
        detections = self.multi_classifier.predict(detections)
        labels = [
            f"#{tracker_id} {results.names[class_id]} {action} #{behavior}#"
            for class_id, tracker_id, action, behavior
            in zip(detections.class_id, detections.tracker_id, detections.data["actions"], detections.data["behaviors"])
        ]
        self.times[1] += time.time() - start_time
        start_time = time.time()
        dict_behaviors_colors = {'normal': '#00FF00', 'abnormal': '#FF0000'}
        annotated_frame = self.pose_annotator.annotate(
            frame.copy(), detections=detections, keypoints=results.keypoints)
        annotated_frame = self.label_annotator.annotate(
            annotated_frame, detections=detections, labels=labels)
        for i, (box, behavior) in enumerate(zip(detections.xyxy, detections.data["behaviors"])):
            color = sv.Color.from_hex(dict_behaviors_colors[behavior])
            single_detection = sv.Detections(
                xyxy=np.array([box]),
                class_id=np.array([0]),  # Добавлен class_id
                tracker_id=np.array([i])  # Добавлен tracker_id
            )
            color_annotator = ColorAnnotator(color=color, opacity=0.4, color_lookup=sv.ColorLookup.TRACK)
            annotated_frame = color_annotator.annotate(
                annotated_frame, detections=single_detection
            )

        self.times[2] += time.time() - start_time

        self.frame_count += 1
        info_dict = {
            "Num_People": len(detections),
            "Actions": detections.data["actions"],
            "Behaviors": detections.data["behaviors"]
        }
        return annotated_frame, info_dict

    def print_average_times(self):
        avg_times = [t / self.frame_count for t in self.times]
        print(f"Average time for model prediction: {avg_times[0]}")
        print(f"Average time for tracking and classification: {avg_times[1]}")
        print(f"Average time for annotation: {avg_times[2]}")


def get_video_length(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Failed to open video file")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames


def process_video(file):
    # Create temporary input file to store the uploaded video
    with tempfile.NamedTemporaryFile(delete=False) as tmp_input_file:
        tmp_input_file.write(file.file.read())
        input_video_path = tmp_input_file.name

    root_dir = os.path.dirname(os.path.abspath(__file__))
    output_video_path = os.path.join(root_dir, "processed_video.mp4")
    processor = VideoProcessor()
    total_frames = get_video_length(input_video_path)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError("Failed to open video file")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    log_entries = []
    timestamp_prev = 0

    with tqdm(total=total_frames) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            result, info_dict = processor.callback(frame, int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
            out.write(result)

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if timestamp - timestamp_prev >= 1:
                log_entry = {
                    "Timestamp": timestamp,
                    "Frame": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                    "Num_People": info_dict["Num_People"],
                    "Actions": info_dict["Actions"],
                    "Behaviors": info_dict["Behaviors"]
                }
                log_entries.append(log_entry)
                timestamp_prev = timestamp

            pbar.update(1)

    cap.release()
    out.release()

    processor.print_average_times()

    # Remove the temporary input file
    os.remove(input_video_path)

    # Convert log entries to JSON
    log_json = json.dumps(log_entries, default=str)

    return output_video_path, log_json


# def process_video_with_progress(source_path, target_path, target_fps=30):
#     cap = cv2.VideoCapture(source_path)
#     if not cap.isOpened():
#         raise ValueError("Failed to open video file")
#
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(target_path, fourcc, target_fps, (width, height))
#
#     frame_interval = int(cap.get(cv2.CAP_PROP_FPS) // target_fps)
#     frame_idx = 0
#
#     with tqdm(total=total_frames // frame_interval) as pbar:
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#
#             if frame_idx % frame_interval == 0:
#                 result = processor.callback(frame, frame_idx)
#                 out.write(result)
#                 pbar.update(1)
#
#             frame_idx += 1
#
#     cap.release()
#     out.release()
#
#     processor.print_average_times()

# video_path = "./test_data/tt.mp4"
# process_video_with_progress(video_path, "outputs/2tt_catboost_BEHAVIORS.mp4")
