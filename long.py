import os
import argparse
import cv2
import numpy as np
import time
import csv
from threading import Thread

class VideoStream:
    def __init__(self, resolution=(800, 480), framerate=30):
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

# Fonction pour gérer les clics de souris
def click_event(event, x, y, flags, param):
    global stop_script
    if event == cv2.EVENT_LBUTTONDOWN:
        # Vérifier si le clic est dans la zone du bouton rouge
        if 10 <= x <= 110 and 10 <= y <= 60:
            stop_script = True

def get_last_frame_count(csv_path):
    if not os.path.exists(csv_path):
        return 0
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
        if len(rows) > 1:
            return int(rows[-1][0])
    return 0

# Initialisation du parser d'arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', default='Sample_TFLite_model')
parser.add_argument('--graph', default='long_quant_edgetpu.tflite')
parser.add_argument('--labels', default='label-long.txt')
parser.add_argument('--threshold', default=0.7)
parser.add_argument('--resolution', default='1280x720')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)

CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

interpreter = Interpreter(model_path=PATH_TO_CKPT,
                          experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

frame_rate_calc = 1
freq = cv2.getTickFrequency()

videostream = VideoStream(resolution=(1280, 720), framerate=30).start()
time.sleep(1)

csv_file_path = 'detection_data.csv'
start_frame_count = get_last_frame_count(csv_file_path)
current_frame = start_frame_count

output_video_name = f"output_{current_frame}.mp4"
output_video = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(*'mp4v'), 13, (1280, 720))

detection_data = []

cv2.namedWindow('Object detector', cv2.WND_PROP_FULLSCREEN)
cv2.resizeWindow('Object detector', 800, 400)

stop_script = False

# Attacher l'événement de clic de souris
cv2.setMouseCallback('Object detector', click_event)

while not stop_script:
    t1 = cv2.getTickCount()

    frame1 = videostream.read()

    display_frame = cv2.resize(frame1, (800, 480))
    display_frame = cv2.rotate(display_frame, cv2.ROTATE_180)
    record_frame = frame1.copy()
    record_frame = cv2.rotate(record_frame, cv2.ROTATE_180)

    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    frame_detections = []
    detections_found = False

    for i in range(len(scores)):
        if scores[i] > min_conf_threshold:
            detections_found = True
            ymin, xmin, ymax, xmax = boxes[i]
            ymin = int(max(1, (ymin * display_frame.shape[0])))
            xmin = int(max(1, (xmin * display_frame.shape[1])))
            ymax = int(min(display_frame.shape[0], (ymax * display_frame.shape[0])))
            xmax = int(min(display_frame.shape[1], (xmax * display_frame.shape[1])))

            cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), (0, 0, 0), 1)

            object_name = labels[int(classes[i])].split(' ')[-1]
            label = '%s: %d%%' % (object_name, int(scores[i] * 100))
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(display_frame, (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10),
                          (0, 0, 0), cv2.FILLED)
            cv2.putText(display_frame, label, (xmin, label_ymin - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            avg_xy = (xmin + ymin + xmax + ymax) / 4
            frame_detections.append([current_frame, int(scores[i] * 100), ymin, xmin, ymax, xmax, avg_xy])

    if not detections_found:
        frame_detections.append([current_frame, 0, 0, 0, 0, 0, 0])

    detection_data.extend(frame_detections)

    current_frame += 1

    cv2.rectangle(display_frame, (10, 10), (110, 60), (0, 0, 255), -1)
    cv2.putText(display_frame, 'stop', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Object detector', display_frame)
    output_video.write(record_frame)

    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1

    if cv2.waitKey(1) == ord('q'):
        break

with open(csv_file_path, 'a', newline='') as file:
    writer = csv.writer(file)
    if start_frame_count == 0:
        writer.writerow(['frames', 'Confidence', 'Ymin', 'Xmin', 'Ymax', 'Xmax', 'x+y'])
    writer.writerows(detection_data)

cv2.destroyAllWindows()
videostream.stop()
output_video.release()
