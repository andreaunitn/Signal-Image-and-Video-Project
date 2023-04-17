from reid.feature_extraction.cnn import extract_cnn_feature
from reid.dist_metric import DistanceMetric
from reid.utils.data import transforms as T
from reid.evaluators import Evaluator
from reid import models
from PIL import Image
import os.path as osp
import numpy as np
import torch
import cv2

def cosine_similarity_torch(x1, x2):

    # Compute the dot product between the input tensors
    dot_product = torch.dot(x1, x2)
    
    # Compute the L2 norm of the input tensors along the last dimension
    x1_norm = torch.norm(x1, p=2, dim=-1)
    x2_norm = torch.norm(x2, p=2, dim=-1)
    
    # Compute the cosine similarity between the input tensors
    cosine_sim = dot_product / (x1_norm * x2_norm)
    
    return cosine_sim

def load_checkpoint(fpath):
    if osp.isfile(fpath):
        checkpoint = torch.load(fpath)
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))

# Load the YOLOv3 model and configuration files
#net = cv2.dnn.readNetFromDarknet('yolo/yolov3.cfg', 'yolo/yolov3.weights')
net = cv2.dnn.readNetFromDarknet('yolo/yolov3-tiny.cfg', 'yolo/yolov3-tiny.weights')

# Load the COCO class labels
classes = []
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Set the input and output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

# Load the video stream
cap = cv2.VideoCapture(0)

# Image counter to be incremented every time an image is saved
counter = 0

# Loading the model
model = models.create("resnet50", num_features=1024, dropout=0, num_classes=751, last_stride=1)
checkpoint = load_checkpoint("model_best.pth.tar")
model.load_state_dict(checkpoint['state_dict'])

# Initializing evaluation metric
metric = DistanceMetric(algorithm="euclidean")

# Initializing evaluator
evaluator = Evaluator(model)

# List af all saved images features
dataset_features = []

# Loop over frames from the video stream
while True:
    # Read the next frame from the video stream
    ret, frame = cap.read()
    frame_copy = frame.copy()

    # If there was an error reading the frame, break out of the loop
    if not ret:
        break

    # Detect people in the frame using YOLOv3
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    boxes = []
    confidences = []
    class_ids = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == 0 and confidence > 0.5:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = center_x - w // 2
                y = center_y - h // 2
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Define image transformation
    height = 256
    width = 128
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])

    # Draw bounding boxes around the people and save cropped images
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(frame_copy, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame_copy, f"{label}: {confidence:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
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
                file_name = f"{label}_{counter}.jpg"
                counter += 1
                image_tensor = crop_img[:, :, ::-1]
                image_tensor = Image.fromarray(image_tensor.astype('uint8'), 'RGB')
                image_tensor = test_transformer(image_tensor)
                image_tensor = image_tensor

                if torch.backends.mps.is_available():
                    model = model.to('mps')
                else:
                    model = model.cuda()

                query_features = extract_cnn_feature(model, image_tensor.unsqueeze(0))

                if counter > 5:

                    # Convert the list of features to a tensor
                    all_features = torch.stack(dataset_features)

                    # Compute the cosine similarity between the query image's feature and all other images' features
                    cos_sim = torch.nn.functional.cosine_similarity(query_features, all_features)

                    # Get the index of the most similar image
                    most_similar_index = torch.argmax(cos_sim)

                    print("{label}_{most_similar_index}.jpg")
                    closest_img = cv2.imread(f"{label}_{most_similar_index}.jpg")
                    cv2.imshow('most_similar_image', closest_img)


                dataset_features += query_features
                cv2.imwrite(file_name, crop_img)
            
    # Show the output frame
    cv2.imshow('frame', frame_copy)

    # Wait for a key press
    key = cv2.waitKey(1)

    # If the 'q' key was pressed, break out of the loop
    if key == ord('q'):
        break

print(dataset_features)

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()