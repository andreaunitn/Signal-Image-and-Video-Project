import cv2

# Load the video stream
cap = cv2.VideoCapture(0)

# Load the pre-trained people detection model
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Loop over frames from the video stream
while True:
    # Read the next frame from the video stream
    ret, frame = cap.read()

    # If there was an error reading the frame, break out of the loop
    if not ret:
        break

    # Detect people in the frame
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

    # Draw bounding boxes around the people
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Wait for a key press
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()