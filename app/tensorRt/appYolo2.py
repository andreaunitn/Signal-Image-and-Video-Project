from reid.feature_extraction.cnn import extract_cnn_feature
from reid.utils.data import transforms as T
import matplotlib.colors as colors
from utils import preproc, vis, customVis
from utils import BaseEngine
from PIL import ImageColor
from reid import models
from PIL import Image
import os.path as osp
import numpy as np
import pickle
import random
import time
import torch
import cv2

from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
from PyQt6.QtGui import QImage, QPixmap, QPalette, QColor
from PyQt6.QtCore import QTimer, Qt, QTime

# ---------------------------------------------------------
# Utility functions
def load_model(fpath):
    if osp.isfile(fpath):
        model = torch.load(fpath)
        print("=> Loaded model '{}'".format(fpath))
        return model
    else:
        raise ValueError("=> No model found at '{}'".format(fpath))
    
def get_color_list():
    color_list = []
    hex_values = list(colors.cnames.values())

    for hex in hex_values:
        color_list.append(ImageColor.getcolor(hex, "RGB"))

    random.shuffle(color_list)
    return color_list
# ---------------------------------------------------------

# Opencv and Yolo
# ---------------------------------------------------------
# Parameters
is_first_frame = True # First frame
dataset_features = [] # List af all saved images features
feature_to_id = {} # Dictionary from feature tensor to id of relative person
new_ids = 0 # Number of new identies
tot_ids = 0 # Total number of identities
threshold = 0.75 # Threshold for new ids
detect = False # At least one person has been detected
detect_counter = 0 # Number of frames for new ids
color_list = get_color_list() # List of colors for bounding boxes
color_counter = 0 # Counter for selecting the boundng box color
person_to_color = {} # Dictionary from person id to the color of the bounding box

# Image transformations
height = 256
width = 128
normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
test_transformer = T.Compose([
    T.RectScale(height, width),
    T.ToTensor(),
    normalizer,
])

## Load the YOLOv7-tiny model and configuration files
#net = cv2.dnn.readNetFromDarknet('yolo/yolov4-tiny.cfg', 'yolo/yolov4-tiny.weights')

#if torch.backends.cudnn.is_available():
#    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

## Load the COCO class labels
#classes = []
#with open('coco.names', 'r') as f:
#    classes = [line.strip() for line in f.readlines()]

## Set the input and output layers
#layer_names = net.getLayerNames()
#output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

# Load the video stream
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Prepare the tensorRt engine
class Predictor(BaseEngine):
    def __init__(self, engine_path):
        super(Predictor, self).__init__(engine_path)
        self.n_classes = 80  # your model classes

pred = Predictor(engine_path="yolov7-tiny.trt")

# loading the model
model = models.create("resnet50", num_features=1024, dropout=0, num_classes=751, last_stride=2)
m = load_model("../model_best.pth.tar")
model.load_state_dict(m['state_dict'])

if torch.backends.mps.is_available():
    model = model.to('mps')
else:
    model = model.cuda()
# ---------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # PyQt 
        # -------------------
        # Read the size of the first frame
        ret, frame = cap.read()
        if not ret:
            exit(1)
        height, width, _ = frame.shape

        # Set the width of the window and the layout to include padding
        padding = 450
        window_width = int(width / 2) + padding

        # Set the background color of the main window to white
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(255, 255, 255))
        self.setPalette(palette)

        layout = QVBoxLayout()

        # Create a label to display the image
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(window_width, int(height / 2))
        layout.addWidget(self.image_label, stretch=2)

        self.image_label.setStyleSheet("color: black;")

        people_layout = QVBoxLayout()

        # Set the width of the labels to include padding
        label_width = int((window_width - padding) / 2)

        self.n_people = QLabel(self)
        self.n_people.setText("Tot. number of people: {}".format(new_ids))
        self.n_people.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.n_people.setStyleSheet("color: black; font-weight: bold; font-size: 20px;")
        self.n_people.setFixedSize(label_width, self.n_people.height())
        people_layout.addWidget(self.n_people)

        self.new_people = QLabel(self)
        self.new_people.setText("No new person identified")
        self.new_people.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.new_people.setStyleSheet("color: black; font-weight: bold; font-size: 20px;")
        self.new_people.setFixedSize(label_width, self.new_people.height())
        people_layout.addWidget(self.new_people)

        layout.addLayout(people_layout)

        # Set the layout of the main window
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Initialize variables for frame rate calculation
        self.frames = 0
        self.fps = 0
        self.start_time = QTime.currentTime()

        # Set up a timer to periodically update the image
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_image)
        self.timer.start(0)
        # -------------------

    def update_image(self):
        global is_first_frame, dataset_features, model, new_ids, tot_ids, detect, detect_counter, color_counter, pred

        label = "person"

        # Read the next frame from the video stream
        ret, frame = cap.read()
        frame_copy = frame.copy()

        # If there was an error reading the frame, break out of the loop
        if not ret:
            assert False, "Error while reading the frame"


        ## Detect people in the frame using YOLO
        #blob = cv2.dnn.blobFromImage(frame, 1/255.0, (256, 256), swapRB=True, crop=False)
        #net.setInput(blob)
        #outputs = net.forward(output_layers)
        #boxes = []
        #confidences = []
        #class_ids = []
        #for output in outputs:
        #    for detection in output:
        #        scores = detection[5:]
        #        class_id = np.argmax(scores)
        #        confidence = scores[class_id]

        #        if class_id == 0 and confidence > 0.5:
        #            center_x = int(detection[0] * frame.shape[1])
        #            center_y = int(detection[1] * frame.shape[0])
        #            w = int(detection[2] * frame.shape[1])
        #            h = int(detection[3] * frame.shape[0])
        #            x = center_x - w // 2
        #            y = center_y - h // 2
        #            boxes.append([x, y, w, h])
        #            confidences.append(float(confidence))
        #            class_ids.append(class_id)

        ## Apply non-maximum suppression to remove overlapping boxes
        #indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        _, indices = pred.customProcessVideo(ret, frame_copy, width, height)


        # Draw bounding boxes around the people and save cropped images
        if len(indices) > 0:
            tot_ids = len(indices)

            for i in range(len(indices)):
                x = indices[i][0]
                y = indices[i][1]
                w = indices[i][2] - x
                h = indices[i][3] - y
                    
                # Crop the image within the bounding box and save it to a file
                if x < 0:
                    w += x
                    x = 0

                if y < 0:
                    h += y
                    y = 0

                if x + w > frame.shape[1]:
                    w = frame.shape[1] - x

                if y + h > frame.shape[0]:
                    h = frame.shape[0] - y

                if w > 0 and h > 0:
                    crop_img = frame[y:y+h, x:x+w]
                    image_tensor = crop_img[:, :, ::-1]
                    image_tensor = Image.fromarray(image_tensor.astype('uint8'), 'RGB')
                    image_tensor = test_transformer(image_tensor)

                    # Use the model to extract people's features
                    if torch.backends.mps.is_available():
                        query_features = extract_cnn_feature(model, image_tensor.unsqueeze(0))
                    else:
                        query_features = extract_cnn_feature(model, image_tensor.unsqueeze(0).cuda())

                    id_in_frame = 0
                    color = (0,0,0)

                    # First frame
                    if is_first_frame:
                        color = color_list[color_counter]
                        person_to_color[new_ids] = color

                        detect = True
                        file_name = f"{label}_{new_ids}.jpg"

                        new_ids += 1
                        color_counter += 1
                    else:
                        all_features = torch.stack(dataset_features)

                        # Compute the cosine similarity between the query image's feature and all other images' features
                        cos_sim = torch.nn.functional.cosine_similarity(query_features, all_features)

                        # Get the index, sim. value and features of the most similar image
                        most_similar_index = torch.argmax(cos_sim)
                        cos_value = cos_sim[most_similar_index].item()
                        most_similar_features = dataset_features[most_similar_index]
                        
                        # New person detected
                        if cos_value < threshold:
                            color = color_list[color_counter]
                            person_to_color[new_ids] = color

                            detect = True
                            id_in_frame = new_ids
                            file_name = f"{label}_{id_in_frame}.jpg"
                            cv2.imwrite(file_name, crop_img)

                            new_ids += 1
                            color_counter += 1
                        else:
                            # No new person detected -> find most similar image
                            id_in_frame = feature_to_id[hash(pickle.dumps(most_similar_features))]
                            closest_img = cv2.imread(f"{label}_{id_in_frame}.jpg")
                            #cv2.imshow('most_similar_image', closest_img)

                            file_name = f"{label}_{id_in_frame}.jpg"
                            detect = False

                            color = person_to_color[id_in_frame]

                    dataset_features += query_features

                    # Add feature to dictionary and save id image
                    feature_to_id[hash(pickle.dumps(query_features[0]))] = id_in_frame
                    if not osp.isfile(file_name):
                        cv2.imwrite(file_name, crop_img)

                    is_first_frame = False
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"Person id: {id_in_frame}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        else:
            tot_ids = 0
            detect = False

        # Update labels
        self.n_people.setText("Tot. number of people: {}".format(tot_ids))
        self.n_people.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.n_people.setStyleSheet("color: black; font-weight: bold; font-size: 20px;")

        if detect:
            detect_counter = 15

        if detect_counter > 0:
                self.new_people.setText("New people detected!")
                self.new_people.setAlignment(Qt.AlignmentFlag.AlignLeft)
                self.new_people.setStyleSheet("color: red; font-weight: bold; font-size: 20px;")
        else:
            self.new_people.setText("No new person identified")
            self.new_people.setAlignment(Qt.AlignmentFlag.AlignLeft)
            self.new_people.setStyleSheet("color: black; font-weight: bold; font-size: 20px;")

        detect_counter -= 1

        # Resize the frame to match the size of the label
        frame = cv2.resize(frame, (self.image_label.width() - 450, self.image_label.height()))

        # Calculate the frame rate
        self.frames += 1
        elapsed_time = self.start_time.msecsTo(QTime.currentTime())
        if elapsed_time > 1000:
            self.fps = self.frames / (elapsed_time / 1000)
            self.frames = 0
            self.start_time = QTime.currentTime()

        # Display the FPS value on the image frame
        fps_text = f"FPS: {self.fps:.2f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Convert the image to a Qt-compatible format
        image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format.Format_BGR888)

        # Display the image on the label
        self.image_label.setPixmap(QPixmap.fromImage(image))

if __name__ == "__main__":

    pred.get_fps()
    video = 0
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()

    # Release the video stream and close all windows
    cap.release()
    cv2.destroyAllWindows()
    