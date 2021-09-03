import argparse
import os
import re
import time

import cv2
import numpy as np

from centroidtracker import CentroidTracker
from utils import detect_objects, load_labels, make_interpreter,parse_lines

#https://stackoverflow.com/questions/54093424/why-is-tensorflow-lite-slower-than-tensorflow-on-desktop


default_model_dir = './models'
default_model = 'ssd_mobilenet_v2_coco_quant_postprocess.tflite'
default_labels = 'coco_labels.txt'

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='.tflite model path',
  default=os.path.join(default_model_dir,default_model))
parser.add_argument('--labels', help='label file path',
  default=os.path.join(default_model_dir, default_labels)),
parser.add_argument('--threshold', type=float, default=0.4,
  help='classifier score threshold')
parser.add_argument('--input', type=str, help='video path', default='../data/test.avi')
parser.add_argument('--gates', type=str, help='gates', default='./gates/test_avi.json')


args = parser.parse_args()

print('Loading {} with {} labels.'.format(args.model, args.labels))

print('Loading file {}.'.format(args.input))
capture = cv2.VideoCapture(args.input)

labels = load_labels(args.labels)
gates = parse_lines(args.gates)

#print(gates)

interpreter = make_interpreter(args.model)
interpreter.allocate_tensors()

ct = CentroidTracker(gates)

if not capture.isOpened():
  print('Unable to open: ' + args.input)
  exit(0)

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    start_time = time.monotonic()
    results = detect_objects(interpreter, frame, args.threshold)
    elapsed_ms = (time.monotonic() - start_time) * 1000

    print("Inference time: {0:.2f} ms".format(elapsed_ms))

    objects = ct.update(results)

    for (objectID, object) in objects.items():
      text = "ID {0} class {1} confidence {2}%".format(objectID,labels.get(object[2],'ND'),object[3])
      cv2.putText(frame, text, (object[1] - 10, object[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
      cv2.circle(frame, (object[1], object[0]), 4, (0, 255, 0), -1)

    cv2.imshow("tracker output", frame)


    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
      break
  
capture.release()
cv2.destroyAllWindows()

