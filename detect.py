import cv2
import numpy as np
import glob
import time

INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.4

# load class names
with open('./classes/yv8s_vehicle_det.txt', 'r') as f:
     class_names = f.read().split('\n')

COLORS = []
with open('./classes/yv8s_vehicle_det_color.txt', 'r') as f:
     linhas = f.read().split('\n')
     for linha in linhas:
         valores = linha.split(',')
         cor = []
         for valor in valores:
             cor.append(int(valor))
         COLORS.append(cor.copy())
         cor.clear()

# load YOLO
net = cv2.dnn.readNetFromONNX('./onnx/yv8s_vehicle_det.onnx')
# ln = model.getLayerNames()
# ln = [ln[i - 1] for i in model.getUnconnectedOutLayers()]

net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# read the image
img = cv2.imread('car.png')
# create blob from image
blob = cv2.dnn.blobFromImage(img, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
# set the blob to th model
net.setInput(blob)
t0 = time.time()
# forward pass through the model to carry out the detection
output = net.forward()
t = time.time()

print(output.shape)
output = output.transpose((0, 2, 1))

numPred = output[0].shape[0]

boxes = []
confs = []
class_ids = []
image_h, image_w = img.shape[:2]
x_factor = image_w / INPUT_WIDTH
y_factor = image_h / INPUT_HEIGHT

for i in range(numPred):
     pred = output[0][i]

     classes_score = pred[4:]
     class_id = np.argmax(classes_score)
     conf = classes_score[class_id]
    
     if (conf > CONFIDENCE_THRESHOLD):
        confs.append(conf)
        label = class_names[int(class_id)]
        class_ids.append(class_id)

        # extract boxes
        x, y, w, h = pred[0].item(), pred[1].item(), pred[2].item(), pred[3].item() 
        left = int((x - 0.5 * w) * x_factor)
        top = int((y - 0.5 * h) * y_factor)
        width = int(w * x_factor)
        height = int(h * y_factor)
        box = np.array([left, top, width, height])
        boxes.append(box)

r_class_ids, r_confs, r_boxes = list(), list(), list()

indexes = cv2.dnn.NMSBoxes(boxes, confs, CONFIDENCE_THRESHOLD, NMS_THRESHOLD) 
for i in indexes:
    r_class_ids.append(class_ids[i])
    r_confs.append(confs[i])
    r_boxes.append(boxes[i])

    (x, y) = (boxes[i][0], boxes[i][1])
    (w, h) = (boxes[i][2], boxes[i][3])

    color = COLORS[class_ids[i]]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    text = "{}: {:.4f}".format(class_names[int(class_ids[i])], confs[i])
    cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

cv2.imshow('Predição', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

