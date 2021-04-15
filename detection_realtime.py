#Import Relevant Libraries
import numpy as np
import cv2
import time
from grabscreen import grab_screen
from tensorflow.keras.models import load_model


def main():
    '''Input size expected by the classification_model'''
    HEIGHT = 32
    WIDTH = 32
    '''Load YOLO (YOLOv3 or YOLOv4-Tiny)'''
    #net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_training.cfg")
    net = cv2.dnn.readNet("yolov4-tiny_training_last.weights", "yolov4-tiny_training.cfg")

    classes = []
    with open("signs.names.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    #get last layers names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    check_time = True
    confidence_threshold = 0.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    start_time = time.time()
    frame_count = 0

    detection_confidence = 0.5
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    '''Load Classification Model'''
    classification_model = load_model('traffic.h5') #load mask detection model
    classes_classification = []
    with open("signs_classes.txt", "r") as f:
        classes_classification = [line.strip() for line in f.readlines()]

    '''To test the AI on the webcam'''
    video_capture = cv2.VideoCapture(0)

    while True:
        '''Test: WEBCAM '''
        #_, img = video_capture.read()
        '''Test: SCREEN '''
        img = grab_screen(region=(200,80,1000,680))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #get image shape
        frame_count +=1
        height, width, channels = img.shape
        window_width = width

        # Detecting objects (YOLO)
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen (YOLO)
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > confidence_threshold:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]]) + "=" + str(round(confidences[i]*100, 2)) + "%"
                cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 2)
                '''crop the detected signs -> input to the classification model'''
                crop_img = img[y:y+h, x:x+w]
                if len(crop_img) >0:
                    crop_img = cv2.resize(crop_img, (WIDTH, HEIGHT))
                    crop_img =  crop_img.reshape(-1, WIDTH,HEIGHT,3)
                    prediction = np.argmax(classification_model.predict(crop_img))
                    label = str(classes_classification[prediction])
                    cv2.putText(img, label, (x, y), font, 0.5, (255,0,0), 2)

        elapsed_time = time.time() - start_time
        fps = frame_count/elapsed_time
        print ("fps: ", str(round(fps, 2)))
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord ('q'):
            break
    cv2.destroyAllWindows()

main()
