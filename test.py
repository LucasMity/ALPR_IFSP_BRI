import cv2
import numpy as np
import glob
import time
from yolo import yoloDetection as yolo

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

ONNXArchive = './onnx/yv8s_char_seg.onnx'
Class_Names = loadClassNames('./classes/char_det.txt')
Class_Colors = loadClassColors('./classes/char_det_color.txt')
arquivo = './IndividualResults/char_seg.txt'

Dataset_dir = './Projeto OCR/Bases_Dados/UFPR-ALPR_charSeg_dataset/images/test/'
images_path = glob.glob(Dataset_dir + '*.png')

test = yolo(INPUT_IMGSZ= 384,
            SCORE_THRESHOLD= 0.25,
            NMS_THRESHOLD= 0.45,
            CLASS_NAMES= Class_Names,
            CLASS_COLORS= Class_Colors,
            ONNXArchive= ONNXArchive)

det = []
for img_path in images_path:
     aux = []
     image_name = img_path[len(Dataset_dir):]
     img = cv2.imread(img_path)
     h, w = img.shape[:2]

     test.runYOLODetection(img)

     for detection in test.DETECTIONS:
          aux.append(image_name)
          aux.append(detection['CLASS_ID'])
          aux.append(detection['BOX'][0] / w)
          aux.append(detection['BOX'][1] / h)
          aux.append(detection['BOX'][2] / w)
          aux.append(detection['BOX'][3] / h)
          try:
               with open(arquivo, "a") as arq:
                    arq.write(f'{aux[0]} {aux[1]} {aux[2]} {aux[3]} {aux[4]} {aux[5]}\n')
          except:
               with open(arquivo, "w") as arq:
                    arq.write(f'{aux[0]} {aux[1]} {aux[2]} {aux[3]} {aux[4]} {aux[5]}\n')
          aux.clear()


