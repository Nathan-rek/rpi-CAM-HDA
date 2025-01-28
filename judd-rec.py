import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import csv

class VideoStream:
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

parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', default='Sample_TFLite_model')
parser.add_argument('--graph', default='judd2_quant_edgetpu.tflite')
parser.add_argument('--labels', default='label-judd.txt')
parser.add_argument('--threshold', default=0.5)
parser.add_argument('--resolution', default='1280x720')
parser.add_argument('--edgetpu', action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

outname = output_details[0]['name']
if 'StatefulPartitionedCall' in outname:
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

frame_rate_calc = 1
freq = cv2.getTickFrequency()

videostream = VideoStream(resolution=(1280, 720), framerate=15).start()
time.sleep(1)

detection_data = []

output_video = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (1280, 720))

cv2.namedWindow('Object detector', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Object detector', 640, 480)

while True:
    t1 = cv2.getTickCount()

    frame1 = videostream.read()
    
    # Frame pour l'affichage (640x480 avec détection)
    display_frame = cv2.resize(frame1, (640, 480))
    
    # Frame pour l'enregistrement (1280x720 sans détection)
    record_frame = frame1.copy()

    # Traitement de la détection sur le frame d'affichage
    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

    frame_detections = []

    for i in range(len(scores)):
        if scores[i] > min_conf_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            ymin = int(max(1, (ymin * display_frame.shape[0])))
            xmin = int(max(1, (xmin * display_frame.shape[1])))
            ymax = int(min(display_frame.shape[0], (ymax * display_frame.shape[0])))
            xmax = int(min(display_frame.shape[1], (xmax * display_frame.shape[1])))

            cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

            object_name = labels[int(classes[i])]
            label = '%s: %d%%' % (object_name, int(scores[i] * 100))
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(display_frame, (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10),
                          (0, 0, 255), cv2.FILLED)
            cv2.putText(display_frame, label, (xmin, label_ymin - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            confidence = int(scores[i] * 100)
            frame_detections.append([confidence, ymin, xmin, ymax, xmax])

    if len(frame_detections) == 0:
        frame_detections.append([0, 0, 0, 0, 0])

    detection_data.append(frame_detections)

    cv2.putText(display_frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    
    # Affichage du frame avec détection
    cv2.imshow('Object detector', display_frame)

    # Enregistrement du frame original sans détection
    output_video.write(record_frame)

    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1

    if cv2.waitKey(1) == ord('q'):
        break

with open('detection_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Confidence', 'Ymin', 'Xmin', 'Ymax', 'Xmax'])
    for frame_detections in detection_data:
        for detection in frame_detections:
            writer.writerow(detection)

cv2.destroyAllWindows()
videostream.stop()
output_video.release()
