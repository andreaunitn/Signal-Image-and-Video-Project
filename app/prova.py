import cv2
from PyQt6.QtCore import QTimer, Qt, QTime
from PyQt6.QtGui import QImage, QPixmap, QPalette, QColor
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Open the image using OpenCV
        self.cap = cv2.VideoCapture(0)

        # Read the size of the first frame
        ret, frame = self.cap.read()
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

        # Set the color of the labels to black
        self.image_label.setStyleSheet("color: black;")

        # Create a QVBoxLayout to hold the two "people" labels
        people_layout = QVBoxLayout()

        # Set the width of the labels to include padding
        label_width = int((window_width - padding) / 2)

        self.n_people = QLabel(self)
        self.n_people.setText("Tot. number of people: {}".format(0))
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

        # Add the QVBoxLayout to the main QVBoxLayout
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

    def update_image(self):
        # Read the image from the camera
        ret, frame = self.cap.read()

        if not ret:
            exit(1)

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
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
    