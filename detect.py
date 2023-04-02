import cv2
import numpy as np
import glob
import time
from yolo import yolo

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

def xywhTOxyixyf(xywh):
     x0, y0 = int(xywh[0] - 0.5 * xywh[2]), int(xywh[1] - 0.5 * xywh[3])
     x1, y1 = int(xywh[0] + 0.5 * xywh[2]), int(xywh[1] + 0.5 * xywh[3])
     return [x0, y0, x1, y1]

def putRectangleDetection(image, xyxy, class_name, score, color, showScore=True):
     cv2.rectangle(image, xyxy[0:2], xyxy[2:], color, 2)
     if showScore:
          text = "{}: {:.4f}".format(class_name, score)
          cv2.putText(image, text, (xyxy[0], xyxy[1] - 5), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1)
     else:
          text = "{}".format(class_name)
          cv2.putText(image, text, (xyxy[0], xyxy[1] - 5), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1)

def cutDetection(image, xyxy):
     return image[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
     
def changeReferenceXY(reference, change):
     change[0] += reference[0]
     change[1] += reference[1]
     change[2] += reference[0]
     change[3] += reference[1]

vehicle_det = yolo(640, 640,
                         0.45, 0.25,
                         loadClassNames('./classes/vehicle_det.txt'),
                         loadClassColors('./classes/vehicle_det_color.txt'),
                         './onnx/yv8s_vehicle_det.onnx')

lp_det_pvd = yolo(640, 640,
                         0.45, 0.25,
                         loadClassNames('./classes/lp_det.txt'),            
                         loadClassColors('./classes/lp_det_color.txt'),
                         './onnx/yv8s_lp_det_pvd.onnx')

char_det = yolo(384, 384,          
                         0.45, 0.25,
                         loadClassNames('./classes/char_det.txt'),
                         loadClassColors('./classes/char_det_color.txt'),
                         './onnx/yv8s_char_det.onnx')


img = cv2.imread('./car2.png')

vehicle_det.runYOLODetection(img)

for i in range(len(vehicle_det.CLASS_IDS)):
     carCoor = xywhTOxyixyf(vehicle_det.BOXES[i])
     lp_det_pvd.runYOLODetection(cutDetection(img, carCoor))
     LPCoor = xywhTOxyixyf(lp_det_pvd.BOXES[i])
     changeReferenceXY(carCoor, LPCoor)

     char_det.runYOLODetection(cutDetection(img, LPCoor))
     
     putRectangleDetection(img, carCoor,
                           vehicle_det.CLASS_NAMES[vehicle_det.CLASS_IDS[i]],
                           vehicle_det.SCORES[i],
                           vehicle_det.CLASS_COLORS[vehicle_det.CLASS_IDS[i]])
     
     putRectangleDetection(img, LPCoor,
                           lp_det_pvd.CLASS_NAMES[lp_det_pvd.CLASS_IDS[i]],
                           lp_det_pvd.SCORES[i],
                           lp_det_pvd.CLASS_COLORS[lp_det_pvd.CLASS_IDS[i]])
     
     for j in range(len(char_det.CLASS_IDS)):
          charCoor = xywhTOxyixyf(char_det.BOXES[j])
          print(char_det.CLASS_NAMES[char_det.CLASS_IDS[j]])
          changeReferenceXY(LPCoor, charCoor)

          putRectangleDetection(img, charCoor,
                           char_det.CLASS_NAMES[char_det.CLASS_IDS[j]],
                           char_det.SCORES[j],
                           char_det.CLASS_COLORS[char_det.CLASS_IDS[j]],
                           showScore= False)


cv2.imshow('Predição', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


