
import cv2
import numpy as np

# Load the YOLOv3 model and configuration files
net = cv2.dnn.readNetFromDarknet('yolo/yolov3.cfg', 'yolo/yolov3.weights')
#net = cv2.dnn.readNetFromDarknet('yolo/yolov3-tiny.cfg', 'yolo/yolov3-tiny.weights')

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
                file_name = f"{label}_{i}_{counter}.jpg"
                counter += 1
                cv2.imwrite(file_name, crop_img)
            

    # Show the output frame
    cv2.imshow('frame', frame_copy)

    # Wait for a key press
    key = cv2.waitKey(1)

    # If the 'q' key was pressed, break out of the loop
    if key == ord('q'):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()

