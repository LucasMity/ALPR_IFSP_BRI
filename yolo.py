import cv2
import numpy as np
import time

class yolo:

    def __init__(self, INPUT_WIDTH, INPUT_HEIGHT,
                    NMS_THRESHOLD, SCORE_THRESHOLD,
                    CLASS_NAMES, CLASS_COLORS,
                    ONNXArchive):
        self.INPUT_WIDTH = INPUT_WIDTH
        self.INPUT_HEIGHT = INPUT_HEIGHT
        self.NMS_THRESHOLD = NMS_THRESHOLD
        self.SCORE_THRESHOLD = SCORE_THRESHOLD
        self.CLASS_NAMES = CLASS_NAMES
        self.CLASS_COLORS = CLASS_COLORS
        self.NET = self.loadYOLOfromONNX(ONNXArchive)
        self.CLASS_IDS = []
        self.SCORES = []
        self.BOXES = []

    def loadYOLOfromONNX(self, ONNXArchive):
        return cv2.dnn.readNetFromONNX(ONNXArchive)

    def setDevice(self, device):
        if device == 'cpu':
            self.NET.setPreferableBackend(cv2.dnn.DNN_BACKEND_CPU)
            self.NET.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        elif device == 'cuda':
            self.NET.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.NET.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def runYOLODetection(self, original_image):
        image_h, image_w = original_image.shape[:2]

        length = max(image_h, image_w)
        image = np.zeros((length, length, 3), np.uint8)
        image[0:image_h, 0:image_w] = original_image

        x_scale = length / self.INPUT_WIDTH
        y_scale = length / self.INPUT_HEIGHT
        # create blob from image
        blob = cv2.dnn.blobFromImage(image, scalefactor= 1/255, size=(self.INPUT_WIDTH, self.INPUT_HEIGHT))
    
        # set the blob to the model
        self.NET.setInput(blob)
        # forward pass through the model to carry out the detection
        outputs = self.NET.forward()

        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []

        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            class_id = np.argmax(classes_scores)
            maxScore = classes_scores[class_id]
            
            if (maxScore >= self.SCORE_THRESHOLD):
                x, y, w, h = outputs[0][i][0], outputs[0][i][1], outputs[0][i][2], outputs[0][i][3]
                
                x = int(x * x_scale)
                y = int(y * y_scale)
                w = int(w * x_scale)
                h = int(h * y_scale)
                box = [x, y, w, h]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(class_id)

        self.CLASS_IDS.clear()
        self.SCORES.clear()
        self.BOXES.clear()

        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, self.SCORE_THRESHOLD, self.NMS_THRESHOLD, 0.5) 
        for i in range(len(result_boxes)):
            index = result_boxes[i]
            self.CLASS_IDS.append(class_ids[index])
            self.SCORES.append(scores[index])
            self.BOXES.append(boxes[index])

