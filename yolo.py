import cv2
import numpy as np
import time

class yolo:
    # Yolo
    INPUT_WIDTH = 640
    INPUT_HEIGHT = 640
    NMS_THRESHOLD = 0.4
    CONFIDENCE_THRESHOLD = 0.4
    CLASS_NAMES = []
    CLASS_COLORS = []
    NET = None

    # Prediction
    CLASS_IDS = []
    CONFS = []
    BOXES = []
    TIME = None

    def __init__(self, INPUT_WIDTH, INPUT_HEIGHT,
                    NMS_THRESHOLD, CONFIDENCE_THRESHOLD,
                    CLASS_NAMES, CLASS_COLORS,
                    ONNXArchive):
        self.INPUT_WIDTH = INPUT_WIDTH
        self.INPUT_HEIGHT = INPUT_HEIGHT
        self.NMS_THRESHOLD = NMS_THRESHOLD
        self.CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD
        self.CLASS_NAMES = CLASS_NAMES
        self.CLASS_COLORS = CLASS_COLORS
        self.loadYOLOfromONNX(ONNXArchive)

    def loadYOLOfromONNX(self, ONNXArchive):
        self.NET = cv2.dnn.readNetFromONNX(ONNXArchive)

    def setDevice(self, device):
        if device == 'cpu':
            self.NET.setPreferableBackend(cv2.dnn.DNN_BACKEND_CPU)
            self.NET.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        elif device == 'cuda':
            self.NET.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.NET.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def runYOLODetection(self, image):
        # create blob from image
        blob = cv2.dnn.blobFromImage(image, 1/255, (self.INPUT_WIDTH, self.INPUT_HEIGHT), swapRB=True, crop=False)
        # set the blob to the model
        self.NET.setInput(blob)
        t0 = time.time()
        # forward pass through the model to carry out the detection
        output = self.NET.forward()
        t = time.time()
        self.TIME = t - t0
        output = output.transpose((0, 2, 1))
        numPred = output[0].shape[0]

        boxes = []
        confs = []
        class_ids = []

        for i in range(numPred):

            pred = output[0][i]
            classes_score = pred[4:]
            class_id = np.argmax(classes_score)
            conf = classes_score[class_id]
            
            if (conf > self.CONFIDENCE_THRESHOLD):
                confs.append(conf)
                class_ids.append(class_id)
                x, y, w, h = pred[0].item(), pred[1].item(), pred[2].item(), pred[3].item()
                box = np.array([x, y, w, h])
                boxes.append(box)

        self.CLASS_IDS.clear()
        self.CONFS.clear()
        self.BOXES.clear()

        indexes = cv2.dnn.NMSBoxes(boxes, confs, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD) 
        for i in indexes:
            self.CLASS_IDS.append(class_ids[i])
            self.CONFS.append(confs[i])
            self.BOXES.append(boxes[i])

