import cv2
import numpy as np
import glob
import time
import yolo

def loadClassNames(ClassArchive):
        with open(ClassArchive, 'r') as f:
            class_names = f.read().split('\n')
        return class_names

def loadClassColors(ClassColorsArchive):
    class_colors = []
    with open(ClassColorsArchive, 'r') as f:
        rows = f.read().split('\n')
        for row in rows:
            values = row.split(',')
            color = []
            for value in values:
                color.append(int(value))
            class_colors.append(color.copy())
            color.clear()
    return class_colors

car_detector = yolo.yolo(640, 640,
                         0.4, 0.4,
                         loadClassNames('./classes/yv8s_vehicle_det.txt'),
                         loadClassColors('./classes/yv8s_vehicle_det_color.txt'),
                         './onnx/yv8s_vehicle_det.onnx')


img = cv2.imread('./car.png')
car_detector.runYOLODetection(img)
print(car_detector.CLASS_IDS)
print(car_detector.CONFS)
print(car_detector.BOXES)
print(car_detector.TIME)

