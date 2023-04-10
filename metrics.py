import cv2
import csv
import numpy as np
import glob
import random
import time
import statistics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt

def loadClassNames(ClassArchive):
        with open(ClassArchive, 'r') as f:
            class_names = f.read().split('\n')
        return class_names

def xywhToxyxy(box):
    return [
        box[0] - box[2] / 2,
        box[1] - box[3] / 2,
        box[0] + box[2] / 2,
        box[1] + box[3] / 2
    ]

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou

def build_confusion_matrix(real, pred, classes):
     y_test = []
     y_pred = []
     labels = classes[:]
     labels.append('Background')
     for image in real:
          for i in range(len(real[image])):
               y_test.append(labels[real[image][i]['class_id']])
               if real[image][i]['pred_index'] == -1:
                    y_pred.append(labels[len(labels)-1])
               else:
                    y_pred.append(labels[pred[image][real[image][i]['pred_index']]['class_id']])
     
     matrix = confusion_matrix(y_test, y_pred, normalize='true', labels=labels)
     dmatrix = ConfusionMatrixDisplay(confusion_matrix= matrix, display_labels=labels)
     fig,ax = plt.subplots(figsize=(20,20))
     plt.rcParams.update({'font.size': 6})
     dmatrix.plot(ax=ax, cmap=plt.cm.Blues, values_format='.1f')
     ticks=np.linspace(0, len(labels)-1,num=len(labels))
     plt.imshow(matrix, interpolation='none', cmap='Blues')
     plt.xticks(ticks,fontsize=8, rotation=45)
     plt.yticks(ticks,fontsize=8)
     plt.grid(False)
     plt.show()


def arqToDict(arquivo):
    Dict = {}
    with open(arquivo, 'r') as arq:
        linhas = arq.readlines()
        for linha in linhas:
             aux = linha.split(' ')
             aux2 = {
                  'class_id': int(aux[1]),
                  'x': float(aux[2]),
                  'y': float(aux[3]),
                  'w': float(aux[4]),
                  'h': float(aux[5])
             }
             try:
                Dict[aux[0]].append(aux2)
             except:
                Dict[aux[0]] = []
                Dict[aux[0]].append(aux2)
             aux.clear()
    return Dict

def calcula_TP_FN_FP(pred, real, classes, IOU_THRESHOLD, compClass=True):
    rel = []
    for clas in classes:
         aux = {
              'class_name' : clas,
              'Qobj': 0,
              'TP': 0,
              'FN': 0,
              'FP': 0
         }
         rel.append(aux)
    rel.append({
         'class_name' : 'all',
         'Qobj': 0,
         'TP': 0,
         'FN': 0,
         'FP': 0 })
    
    for image in real:
         for i in range(len(real[image])):
              rel[real[image][i]['class_id']]['Qobj'] += 1
              rel[len(rel)-1]['Qobj'] += 1
              if real[image][i]['pred_index'] == -1:
                    rel[len(rel)-1]['FN'] += 1
                    rel[real[image][i]['class_id']]['FN'] += 1
              elif compClass:
                    if pred[image][ real[image][i]['pred_index'] ]['class_id'] != real[image][i]['class_id']:
                              rel[len(rel)-1]['FN'] += 1
                              rel[len(rel)-1]['FP'] += 1
                              rel[real[image][i]['class_id']]['FN'] += 1
                              rel[pred[image][ real[image][i]['pred_index'] ]['class_id']]['FP'] += 1
                    elif pred[image][ real[image][i]['pred_index'] ]['class_id'] == real[image][i]['class_id']:
                              if real[image][i]['IOU'] >= IOU_THRESHOLD:
                                   rel[len(rel)-1]['TP'] += 1
                                   rel[real[image][i]['class_id']]['TP'] += 1
                              else:
                                   rel[len(rel)-1]['FP'] += 1
                                   rel[real[image][i]['class_id']]['FP'] += 1
              else:
                   if real[image][i]['IOU'] >= IOU_THRESHOLD:
                         rel[len(rel)-1]['TP'] += 1
                         rel[real[image][i]['class_id']]['TP'] += 1
                   else:
                         rel[len(rel)-1]['FP'] += 1
                         rel[real[image][i]['class_id']]['FP'] += 1
                   

    for image in pred:
        for i in range(len(pred[image])):
             if pred[image][i]['real_index'] == -1:
                  rel[pred[image][i]['class_id']]['FP'] += 1
                  rel[len(rel)-1]['FP'] += 1
            
    return rel
                       
def calcula_PR(rel):
     for r in range(len(rel)):
          rel[r]['precision'] = rel[r]['TP'] / max(1, (rel[r]['TP'] + rel[r]['FP']))
          rel[r]['recall'] = rel[r]['TP'] / max(1, (rel[r]['TP'] + rel[r]['FN']))

def printDadosPR(rel):
     print('class\tQobj\tTP\tFN\tFP\tprec\trecall')
     for r in rel:
          print('{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}'.format(r['class_name'], r['Qobj'], r['TP'], r['FN'], r['FP'], r['precision'], r['recall']))

def placas_det(real, pred):
     det = [0, 0, 0, 0, 0, 0, 0, 0]
     for image in real:
          wrong = 0
          for char in real[image]:
               if char['pred_index'] == -1:
                    wrong += 1
               elif pred[image][ char['pred_index'] ]['class_id'] != char['class_id']:
                    wrong += 1
          det[wrong] += 1
     total = sum(det)
     for d in range(1,len(det)):
          det[d] += det[d-1]
     for d in range(len(det)):
          det[d] /= total
     return det


def corrige_0O(real):
     for image in real:
          for i in range(3):
               if real[image][i]['class_id'] == 0:
                    real[image][i]['class_id'] = 35


predict_file = './test_predicts/predicts2/char_det.txt'
real_file = './real/Rchar_det.txt'
classes = loadClassNames('./classes/char_det.txt')
names = ['image_name', 'class_id', 'x', 'y', 'w', 'h']

predict = arqToDict(predict_file)
real = arqToDict(real_file)

corrige_0O(real)

for index in real:
     for i in range(len(real[index])):
          real[index][i]['IOU'] = 0
          real[index][i]['pred_index'] = -1

for index in predict:
     for i in range(len(predict[index])):
          predict[index][i]['IOU'] = 0
          predict[index][i]['real_index'] = -1

# for image in predict:
#     for i in range(len(predict[image])):
#         for j in range(len(real[image])):
#             iou = bb_intersection_over_union(xywhToxyxy((predict[image][i]['x'], predict[image][i]['y'], predict[image][i]['w'], predict[image][i]['h'])),
#                                              xywhToxyxy((real[image][j]['x'], real[image][j]['y'], real[image][j]['w'], real[image][j]['h'])) )
#             if iou > real[image][j]['IOU']:
#                     real[image][j]['IOU'] = iou
#                     if real[image][j]['pred_index'] != -1:
#                          predict[image][real[image][j]['pred_index']]['IOU'] = 0
#                          predict[image][real[image][j]['pred_index']]['real_class_id'] = -1
#                     real[image][j]['pred_index'] = i
#                     predict[image][i]['IOU'] = iou
#                     predict[image][i]['real_class_id'] = real[image][j]['class_id']

for image in real:
    if image not in predict:
             continue
    for j in range(len(real[image])):
        maxIOU = 0
        maxIndex = -1
        for i in range(len(predict[image])):
            iou = bb_intersection_over_union(xywhToxyxy((predict[image][i]['x'], predict[image][i]['y'], predict[image][i]['w'], predict[image][i]['h'])),
                                             xywhToxyxy((real[image][j]['x'], real[image][j]['y'], real[image][j]['w'], real[image][j]['h'])) )
            if iou > maxIOU:
                 maxIOU = iou
                 maxIOUindex = i

        if iou > predict[image][maxIOUindex]['IOU']:
             if predict[image][maxIOUindex]['real_index'] != -1:
                  real[image][predict[image][maxIOUindex]['real_index']]['IOU'] = 0
                  real[image][predict[image][maxIOUindex]['real_index']]['pred_index'] = -1          
             real[image][j]['IOU'] = iou
             real[image][j]['pred_index'] = maxIOUindex
             predict[image][maxIOUindex]['IOU'] = iou
             predict[image][maxIOUindex]['real_index'] = j
                    



# for i in predict:
#      for j in predict[i]:
#           if j['class_id'] == 24:
#                print(j['class_id'], j['IOU'], j['real_class_id'], i)
# for i in real:
#      for j in real[i]:
#           if j['class_id'] == 0:
#                try :
#                     print(j['class_id'], j['IOU'], predict[i][j['pred_index']]['class_id'])
#                except:
#                     print(j['class_id'], j['IOU'], -1)

p = calcula_TP_FN_FP(predict, real, classes, 0, compClass=True)
calcula_PR(p)
printDadosPR(p)
build_confusion_matrix(real, predict, classes)

print('Placas Totalmente reconhecidas:', placas_det(real, predict))

# print((25 * (len(conjuntos) + 1)) * '=')
# print('{:^25}'.format(''), end = '')
# for conjunto in conjuntos:
#     print('{:^25}'.format(conjunto.upper()), end='')
# print('')
# print((25 * (len(conjuntos) + 1)) * '=')
# print('{:^25}'.format('IOU MÉDIO'), end='')
# for resultado in resultados:
#     print('{:^25}'.format(sum(resultado) / len(resultado)), end='')
# print('')
# print('{:^25}'.format('DESVIO PADRÃO'), end='')
# for resultado in resultados:
#     print('{:^25}'.format(statistics.pstdev(resultado)), end='')
# print('')
# recall, precision = recall_precision(iou, 0.5, predict)
# print('{:^25}'.format('% RECALL'), end='')
# for resultado in resultados:
#     print('{:^25}'.format(recall), end='')
# print('')
# print('{:^25}'.format('% PRECISION'), end='')
# for resultado in resultados:
#     print('{:^25}'.format(precision), end='')
# print('')
# print((25 * (len(conjuntos) + 1)) * '=')
