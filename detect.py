import cv2
import numpy as np
import glob
import time
from yolo import yoloDetection as yoloDet
from yolo import yoloCLS as yoloCLS
from statistics import median

def heuristica_placaBR(detections, classes):
    caracter = {}
    for i in range(len(classes)):
          caracter[classes[i]] = i
    
     # Heurística Conversora para letras
    letra = {'0':'O', '1':'I', '2':'Z', '3':'3', '4':'A', '5':'S', '6':'G', '7':'Z', '8':'B', '9':'9',
             'A':'A', 'B':'B', 'C':'C', 'D':'D', 'E':'E', 'F':'F', 'G':'G', 'H':'H', 'I':'I',
             'J':'J', 'K':'K', 'L':'L', 'M':'M', 'N':'N', 'O':'O', 'P':'P', 'Q':'Q', 'R':'R',
             'S':'S', 'T':'T', 'U':'U', 'V':'V', 'W':'W', 'X':'X', 'Y':'Y', 'Z':'Z'}
    
    # Heurística Conversora para números
    num = { '0':'0', '1':'1', '2':'2', '3':'3', '4':'4', '5':'5', '6':'6', '7':'7', '8':'8', '9':'9',
            'A':'4', 'B':'8', 'C':'C', 'D':'0', 'E':'E', 'F':'F', 'G':'0', 'H':'H', 'I':'1',
            'J':'1', 'K':'K', 'L':'L', 'M':'M', 'N':'N', 'O':'0', 'P':'P', 'Q':'0', 'R':'R',
            'S':'5', 'T':'T', 'U':'U', 'V':'V', 'W':'W', 'X':'X', 'Y':'Y', 'Z':'7'}
    
    if len(detections) == 7:
     for i in range(len(detections)):
               if i < 3: # Letras
                    detections[i]['CLASS_NAME'] = letra[ detections[i]['CLASS_NAME'] ] #Heuristica de Letra
                    detections[i]['CLASS_ID'] = caracter[ detections[i]['CLASS_NAME'] ] #Atualiza CLASS_ID
               else: # Número
                    detections[i]['CLASS_NAME'] = num[ detections[i]['CLASS_NAME'] ] #Heuristica de Número
                    detections[i]['CLASS_ID'] = caracter[ detections[i]['CLASS_NAME'] ] #Atualiza CLASS_ID
          

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

def sortMotLP(listaDic):
     superior = []
     inferior = []
     y = []
     for detection in listaDic:
          y.append(detection['BOX'][1])
     lim = median(y)

     for detection in listaDic:
          if detection['BOX'][1] < lim:
               superior.append(detection)
          else:
               inferior.append(detection)

     superior = sorted(superior, key= lambda dici: dici['BOX'][0])
     inferior = sorted(inferior, key= lambda dici: dici['BOX'][0])

     return superior + inferior
     
def changeReferenceXY(reference, change):
     change[0] += reference[0]
     change[1] += reference[1]
     change[2] += reference[0]
     change[3] += reference[1]

def n_max_conf(n, detections):
     detections_copy = detections[:]
     det = []
     while(len(det) < n and len(detections_copy) > 0):
               remove = 0
               maior = detections_copy[0]['SCORE']

               for i in range(len(detections_copy)):
                    if detections_copy[i]['SCORE'] > maior:
                         maior = detections_copy[i]['SCORE']
                         remove = i
               det.append(detections_copy[remove])
               del detections_copy[remove]
     return det

def extrai_placa(detections):
      placa = ''
      for det in detections:
            placa += det['CLASS_NAME']
      return placa

vehicle_det = yoloDet( 640,
                    0.25, 0.45,
                    loadClassNames('./classes/coco_det.txt'),
                    loadClassColors('./classes/coco_det_color.txt'),
                    './onnx/yv8s_coco.onnx')

lp_det = yoloDet(  640,
                    0.001, 0.45,
                    loadClassNames('./classes/lp_det.txt'),            
                    loadClassColors('./classes/lp_det_color.txt'),
                    './onnx/yv8s_lp_det_pvd.onnx')

char_det = yoloDet(    384,          
                    0.001, 0.45,
                    loadClassNames('./classes/char_det.txt'),
                    loadClassColors('./classes/char_det_color.txt'),
                    './onnx/yv8s_char_det.onnx')


img = cv2.imread('./car2.png')

vehicle_det.runYOLODetection(img)

for i in range(len(vehicle_det.DETECTIONS)):
     if vehicle_det.DETECTIONS[i]['CLASS_NAME'] not in ('car', 'motorcycle'):
           continue
     carCoor = xywhTOxyixyf(vehicle_det.DETECTIONS[i]['BOX'])
     lp_det.runYOLODetection(cutDetection(img, carCoor))
     
     if len(lp_det.DETECTIONS) == 0:
               continue
     lp_det.DETECTIONS = n_max_conf(1, lp_det.DETECTIONS)

     LPCoor = xywhTOxyixyf(lp_det.DETECTIONS[0]['BOX'])
     changeReferenceXY(carCoor, LPCoor)

     char_det.runYOLODetection(cutDetection(img, LPCoor))
     char_det.DETECTIONS = n_max_conf(7, char_det.DETECTIONS)
     
     for x in char_det.DETECTIONS:
           print(x['CLASS_NAME'], end='')
     print('')
     if vehicle_det.DETECTIONS[i]['CLASS_NAME'] == 'car':
               char_det.DETECTIONS = sorted(char_det.DETECTIONS, key= lambda dici: dici['BOX'][0])
     elif vehicle_det.DETECTIONS[i]['CLASS_NAME'] == 'motorcycle':
               char_det.DETECTIONS = sortMotLP(char_det.DETECTIONS)
     
     heuristica_placaBR(char_det.DETECTIONS, char_det.CLASS_NAMES)

     for x in char_det.DETECTIONS:
           print(x['CLASS_NAME'], end='')
     print('')
     
     putRectangleDetection(img, carCoor,
                           vehicle_det.DETECTIONS[i]['CLASS_NAME'],
                           vehicle_det.DETECTIONS[i]['SCORE'],
                           vehicle_det.CLASS_COLORS[vehicle_det.DETECTIONS[i]['CLASS_ID']],
                           showScore=False)
     
     putRectangleDetection(img, LPCoor,
                         #   lp_det.DETECTIONS[0]['CLASS_NAME'],
                           extrai_placa(char_det.DETECTIONS),
                           lp_det.DETECTIONS[0]['SCORE'],
                           lp_det.CLASS_COLORS[lp_det.DETECTIONS[0]['CLASS_ID']],
                           showScore=False)
     
     for j in range(len(char_det.DETECTIONS)):
          charCoor = xywhTOxyixyf(char_det.DETECTIONS[j]['BOX'])
          changeReferenceXY(LPCoor, charCoor)

          print(char_det.DETECTIONS[j]['CLASS_NAME'])

          putRectangleDetection(img, charCoor,
                              char_det.DETECTIONS[j]['CLASS_NAME'],
                              char_det.DETECTIONS[j]['SCORE'],
                              char_det.CLASS_COLORS[char_det.DETECTIONS[j]['CLASS_ID']],
                              showScore= False)


cv2.imshow('Predição', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


