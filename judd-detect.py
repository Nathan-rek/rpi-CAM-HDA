import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import csv  # Pour écrire dans un fichier CSV

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
class VideoStream:
    """Camera object that controls video streaming from the webcam"""
    def __init__(self, resolution=(640, 480), framerate=30):
        self.stream = cv2.VideoCapture(0)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.stream.set(3, resolution[0])
        self.stream.set(4, resolution[1])
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

# Argument parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder containing the model .tflite file(default: Sample_TFLite_model)', default='Sample_TFLite_model')
parser.add_argument('--graph', help='Name of the model .tflite file (default: detect.tflite)', default='judd2_quant_edgetpu.tflite')
parser.add_argument('--labels', help='Name of the labelmap file (default: labelmap.txt)', default='label-judd.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects', default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH', default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection', action='store_true')

args = parser.parse_args()

# Variables for model, labels, and configuration
MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Import TensorFlow Lite libraries
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# Assign filename for Edge TPU model
if use_TPU and GRAPH_NAME == 'detect.tflite':
    GRAPH_NAME = 'edgetpu.tflite'

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Fix for label map if using COCO model
if labels[0] == '???':
    del(labels[0])

# Load TensorFlow Lite model
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

# Input normalization parameters
input_mean = 127.5
input_std = 127.5

# Check output layer name
outname = output_details[0]['name']
if 'StatefulPartitionedCall' in outname:  # TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:  # TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
time.sleep(1)

# Liste pour stocker les données de détection
detection_data = []

# Créer un objet VideoWriter pour enregistrer la vidéo MP4
output_video = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (imW, imH))

while True:
    # Start timer
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = videostream.read()

    # Process frame
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Run model inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get detection results
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

    # List to store detection results for the current frame
    frame_detections = []

    # Draw boxes and labels on the frame
    for i in range(len(scores)):
        if scores[i] > min_conf_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            ymin = int(max(1, (ymin * imH)))
            xmin = int(max(1, (xmin * imW)))
            ymax = int(min(imH, (ymax * imH)))
            xmax = int(min(imW, (xmax * imW)))

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 0), 1)

            object_name = labels[int(classes[i])]
            label = '%s: %d%%' % (object_name, int(scores[i] * 100))
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10),
                          (0, 0, 0), cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

            # Ajouter le pourcentage de confiance et les coordonnées dans le tableau
            confidence = int(scores[i] * 100)
            frame_detections.append([confidence, ymin, xmin, ymax, xmax])

    # Si aucune détection, ajouter une ligne avec des zéros
    if len(frame_detections) == 0:
        frame_detections.append([0, 0, 0, 0, 0])

    # Ajouter les résultats de la frame à la liste globale
    detection_data.append(frame_detections)

    # Display frame rate and image
    cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Object detector', frame)

    # Écrire la frame dans le fichier vidéo MP4
    output_video.write(frame)

    # Calculate frame rate
    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Sauvegarder les données de détection dans un fichier CSV
with open('detection_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Confidence', 'Ymin', 'Xmin', 'Ymax', 'Xmax'])  # En-tête du fichier CSV
    for frame_detections in detection_data:
        for detection in frame_detections:
            writer.writerow(detection)

# Cleanup
cv2.destroyAllWindows()
videostream.stop()
output_video.release()  # Fermer le fichier vidéo MP4
