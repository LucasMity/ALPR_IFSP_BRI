import cv2
import numpy as np
import time

class yoloDetection:

    def __init__(self, INPUT_IMGSZ,
                    SCORE_THRESHOLD,
                    NMS_THRESHOLD,
                    CLASS_NAMES,
                    CLASS_COLORS,
                    ONNXArchive):
        self.INPUT_IMGSZ = INPUT_IMGSZ
        self.SCORE_THRESHOLD = SCORE_THRESHOLD
        self.NMS_THRESHOLD = NMS_THRESHOLD
        self.CLASS_NAMES = CLASS_NAMES
        self.CLASS_COLORS = CLASS_COLORS
        self.NET = self.loadYOLOfromONNX(ONNXArchive)
        self.DETECTIONS = []

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

        scale = length / self.INPUT_IMGSZ
        # create blob from image
        blob = cv2.dnn.blobFromImage(image, scalefactor= 1/255, size=(self.INPUT_IMGSZ, self.INPUT_IMGSZ))
        # r = blob[0, 0, :, :]
        # cv2.imshow('teste.jpg', r)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    
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
            # All Scores
            classes_scores = outputs[0][i][4:]
            # Max score index
            class_id = np.argmax(classes_scores)
            # Max score value
            maxScore = classes_scores[class_id]
            
            # Score filter/Threshold
            if (maxScore >= self.SCORE_THRESHOLD):
                box = [
                    int(outputs[0][i][0] * scale), # X
                    int(outputs[0][i][1] * scale), # Y
                    int(outputs[0][i][2] * scale), # W
                    int(outputs[0][i][3] * scale)  # H
                ]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(class_id)

        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, self.SCORE_THRESHOLD, self.NMS_THRESHOLD, 0.5)

        self.DETECTIONS.clear()

        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]
            detection = {
                'CLASS_ID': class_ids[index],
                'CLASS_NAME': self.CLASS_NAMES[class_ids[index]],
                'SCORE': scores[index],
                'BOX': box,
                'SCALE': scale
            }
            self.DETECTIONS.append(detection)


class yoloCLS:

    def __init__(self, INPUT_IMGSZ,
                    CLASS_NAMES,
                    CLASS_COLORS,
                    ONNXArchive):
        self.INPUT_IMGSZ = INPUT_IMGSZ
        self.CLASS_NAMES = CLASS_NAMES
        self.CLASS_COLORS = CLASS_COLORS
        self.NET = self.loadYOLOfromONNX(ONNXArchive)
        self.CLS = None

    def loadYOLOfromONNX(self, ONNXArchive):
        return cv2.dnn.readNetFromONNX(ONNXArchive)

    def setDevice(self, device):
        if device == 'cpu':
            self.NET.setPreferableBackend(cv2.dnn.DNN_BACKEND_CPU)
            self.NET.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        elif device == 'cuda':
            self.NET.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.NET.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def runYOLOCLS(self, original_image):
        image_h, image_w = original_image.shape[:2]

        length = max(image_h, image_w)
        image = np.zeros((length, length, 3), np.uint8)
        image[0:image_h, 0:image_w] = original_image

        scale = length / self.INPUT_IMGSZ
        # create blob from image
        blob = cv2.dnn.blobFromImage(image, scalefactor= 1/255, size=(self.INPUT_IMGSZ, self.INPUT_IMGSZ))
    
        # set the blob to the model
        self.NET.setInput(blob)
        # forward pass through the model to carry out the detection
        outputs = self.NET.forward()

        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        # All Scores
        classes_scores = outputs[0][0][:]
        # Max score index
        class_id = np.argmax(classes_scores)
        # Max score value
        maxScore = classes_scores[class_id]
        classification = {
            'CLASS_ID': class_id,
            'CLASS_NAME': self.CLASS_NAMES[class_id],
            'SCORE': maxScore
        }
        self.CLS = classification