import cv2
import numpy as np
import glob
import time
from yolo import yoloDetection as yolo
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

def xyxyTOxywh(xyxy):
     w, h = int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])
     x, y = int(xyxy[0] + (w / 2)), int(xyxy[1] + (h / 2))
     return [x, y, w, h]

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

arquivo_1 = './test_predicts/predicts3/vehicle_det.txt'
arquivo_2 = './test_predicts/predicts3/lp_det.txt'
arquivo_3 = './test_predicts/predicts3/char_det.txt'
arquivo_4 = './test_predicts/predicts3/final_lp.txt'

Dataset_dir = 'C:/Users/lucas/OneDrive - ifsp.edu.br/Projeto OCR/Bases_Dados/UFPR-ALPR dataset/images/test/'
images_path = glob.glob(Dataset_dir + '*.png')

vehicle_det = yolo(INPUT_IMGSZ= 640,
            SCORE_THRESHOLD= 0.25,
            NMS_THRESHOLD= 0.45,
            CLASS_NAMES= loadClassNames('./classes/coco_det.txt'),
            CLASS_COLORS= loadClassColors('./classes/coco_det_color.txt'),
            ONNXArchive= './onnx/yv8s_coco.onnx')

lp_det = yolo(INPUT_IMGSZ= 640,
            SCORE_THRESHOLD= 0.001,
            NMS_THRESHOLD= 0.45,
            CLASS_NAMES= loadClassNames('./classes/lp_det.txt'),
            CLASS_COLORS= loadClassColors('./classes/lp_det_color.txt'),
            ONNXArchive= './onnx/yv8s_lp_det_pvd.onnx')

char_det = yolo(INPUT_IMGSZ= 384,
            SCORE_THRESHOLD= 0.001,
            NMS_THRESHOLD= 0.45,
            CLASS_NAMES= loadClassNames('./classes/char_det.txt'),
            CLASS_COLORS= loadClassColors('./classes/char_det_color.txt'),
            ONNXArchive= './onnx/yv8s_char_det.onnx')
det = []
tam = len(images_path)
cont = 0
passou =False
for img_path in images_path:
     aux = []
     cont+=1
     print('PROGRESSO: {:.2%}'.format(cont/tam))
     image_name = img_path[len(Dataset_dir):]
     print(image_name)
     if image_name != 'track0136[12].png' and not passou:
          continue
     else:
          passou = True
     
     img = cv2.imread(img_path)
     h, w = img.shape[:2]

     vehicle_det.runYOLODetection(img)
     while True:
          excluir = -1
          for i in range(len(vehicle_det.DETECTIONS)):
               if vehicle_det.DETECTIONS[i]['CLASS_NAME'] in ('car', 'motorcycle'):
                    if vehicle_det.DETECTIONS[i]['CLASS_NAME'] == 'car':
                         vehicle_det.DETECTIONS[i]['CLASS_ID'] = 0
                    else:
                         vehicle_det.DETECTIONS[i]['CLASS_ID'] = 1
               else:
                    excluir = i
          if excluir != -1:
               del vehicle_det.DETECTIONS[excluir]
          else:
               break


     for detection in vehicle_det.DETECTIONS:    
          aux.append(image_name)
          aux.append(detection['CLASS_ID'])
          aux.append(detection['BOX'][0] / w)
          aux.append(detection['BOX'][1] / h)
          aux.append(detection['BOX'][2] / w)
          aux.append(detection['BOX'][3] / h)
          try:
               with open(arquivo_1, "a") as arq:
                    arq.write(f'{aux[0]} {aux[1]} {aux[2]} {aux[3]} {aux[4]} {aux[5]}\n')
          except:
               with open(arquivo_1, "w") as arq:
                    arq.write(f'{aux[0]} {aux[1]} {aux[2]} {aux[3]} {aux[4]} {aux[5]}\n')
          aux.clear()

     for vehicle in vehicle_det.DETECTIONS:
          vehicle_coor = xywhTOxyixyf(vehicle['BOX'])
          lp_det.runYOLODetection(cutDetection(img, vehicle_coor))
          
          det = []
          if len(lp_det.DETECTIONS) == 0:
               continue

          lp_det.DETECTIONS = n_max_conf(1, lp_det.DETECTIONS)
          
          detection = lp_det.DETECTIONS[0]
          lp_coor = xywhTOxyixyf(detection['BOX'])  
          changeReferenceXY(vehicle_coor, lp_coor)
          lp_coor_xywh = xyxyTOxywh(lp_coor)
          aux.append(image_name)
          aux.append(detection['CLASS_ID'])
          aux.append(lp_coor_xywh[0] / w)
          aux.append(lp_coor_xywh[1] / h)
          aux.append(lp_coor_xywh[2] / w)
          aux.append(lp_coor_xywh[3] / h)
          try:
               with open(arquivo_2, "a") as arq:
                    arq.write(f'{aux[0]} {aux[1]} {aux[2]} {aux[3]} {aux[4]} {aux[5]}\n')
          except:
               with open(arquivo_2, "w") as arq:
                    arq.write(f'{aux[0]} {aux[1]} {aux[2]} {aux[3]} {aux[4]} {aux[5]}\n')
          aux.clear()

          char_det.runYOLODetection(cutDetection(img, lp_coor))

          char_det.DETECTIONS = n_max_conf(7, char_det.DETECTIONS)
         
          if vehicle['CLASS_ID'] == 0:
               char_det.DETECTIONS = sorted(char_det.DETECTIONS, key= lambda dici: dici['BOX'][0])
          elif vehicle['CLASS_ID'] == 1:
               char_det.DETECTIONS = sortMotLP(char_det.DETECTIONS)

          heuristica_placaBR(char_det.DETECTIONS, char_det.CLASS_NAMES)

          for detection in char_det.DETECTIONS:
               char_coor = xywhTOxyixyf(detection['BOX'])  
               changeReferenceXY(lp_coor, char_coor)
               char_coor_xywh = xyxyTOxywh(char_coor)
               aux.append(image_name)
               aux.append(detection['CLASS_ID'])
               aux.append(char_coor_xywh[0] / w)
               aux.append(char_coor_xywh[1] / h)
               aux.append(char_coor_xywh[2] / w)
               aux.append(char_coor_xywh[3] / h)
               try:
                    with open(arquivo_3, "a") as arq:
                         arq.write(f'{aux[0]} {aux[1]} {aux[2]} {aux[3]} {aux[4]} {aux[5]}\n')
               except:
                    with open(arquivo_3, "w") as arq:
                         arq.write(f'{aux[0]} {aux[1]} {aux[2]} {aux[3]} {aux[4]} {aux[5]}\n')
               aux.clear()
          placa = ''
          for detection in char_det.DETECTIONS:
               placa += detection['CLASS_NAME']
          
          try:
               with open(arquivo_4, "a") as arq:
                    arq.write(f'{image_name} {placa}\n')
          except:
               with open(arquivo_4, "w") as arq:
                    arq.write(f'{image_name} {placa}\n')

         