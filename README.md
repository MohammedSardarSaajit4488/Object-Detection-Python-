# Real-Time Object Detection with MobileNet and SSD

This project demonstrates real-time object detection using the MobileNet Single Shot Detector (SSD) model. MobileNet is a lightweight, fast, and accurate model suitable for mobile devices, while SSD is a single-shot detector that enables real-time object detection [1].

## Overview

The system captures video frames from a webcam, performs object detection using the MobileNet SSD model, and displays the resulting frames with bounding boxes and class labels [1].

## Requirements

To run this project, you need the following:

-   A computer or mobile device with a webcam or video input [1]
-   OpenCV (cv2)
-   Numpy

Ensure you have the following files in the same directory as the Python script [1]:

-   `MobileNetSSD.txt`: The Caffe model architecture file.
-   `MobileNetSSD_deploy.caffemodel`: The pre-trained model weights file.

### Downloading Required Files

The `MobileNetSSD.txt` file can be downloaded from:

[https://github.com/chuanqi305/MobileNet-SSD/blob/master/voc/MobileNetSSD_deploy.prototxt](https://github.com/chuanqi305/MobileNet-SSD/blob/master/voc/MobileNetSSD_deploy.prototxt) [1]

## Installation

1.  Install the necessary libraries:

    ```
    pip install opencv-python numpy
    ```
2.  Download the `MobileNetSSD.txt` and `MobileNetSSD_deploy.caffemodel` files and place them in the same directory as your Python script [1].

## Usage

1.  **Run the Python script:**

    Use the following command to execute the Python file:

    ```
    python your_script_name.py --prototxt MobileNetSSD.txt --model MobileNetSSD_deploy.caffemodel
    ```

    Replace `your_script_name.py` with the actual name of your Python script [1].

2.  **Command-Line Arguments:**

    -   `--prototxt`: Path to the `MobileNetSSD.txt` file.
    -   `--model`: Path to the `MobileNetSSD_deploy.caffemodel` file.

    If no parameters are loaded, the script will default to video input from the webcam [4].

3.  **To quit the application**, press `'q'` [1].

## Code Explanation

The Python script performs the following steps:

1.  **Import Libraries:**
    ```
    import numpy as np
    import cv2
    ```
    This imports the necessary libraries, including NumPy for numerical operations and OpenCV for computer vision tasks [2].

2.  **Define Paths and Parameters:**
    ```
    image_path = 'models/roompeople.jpg'
    prototxt_path = 'models/MobileNetSSD_deploy.prototxt.txt'
    model_path = 'models/MobileNetSSD_deploy.caffemodel'
    min_confidence = 0.2
    ```
    This section defines the paths to the image, model architecture, and model weights, as well as the minimum confidence threshold for object detection [2].

3.  **Define Class Labels:**
    ```
    classes = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor", "mobile", "laptop", "mouse",
               "keyboard", "remote", "book", "clock", "vase", "scissors",
               "teddy bear", "hair drier", "toothbrush", "pen", "pencil",
               "notebook", "tablet", "smartphone", "camera", "printer",
               "speaker", "headphones", "microphone", "watch", "wallet",
               "bag", "shoes", "socks", "hat", "glasses", "umbrella",
               "jacket", "shirt", "pants", "shorts", "skirt", "dress",
               "tie", "belt", "ring", "necklace", "bracelet", "earrings",
               "sunglasses", "gloves", "scarf", "cap", "helmet", "bicycle",
               "motorcycle", "car", "bus", "train", "airplane", "boat",
               "ship", "submarine", "rocket", "satellite", "drone",
               "robot", "alien", "monster", "ghost", "zombie", "vampire",
               "werewolf", "dragon", "unicorn", "phoenix", "griffin",
               "mermaid", "centaur", "minotaur", "sphinx", "goblin",
               "troll", "elf", "dwarf", "orc", "fairy", "witch", "wizard",
               "game console", "smartwatch", "e-reader", "VR headset", "mobile","phone"]
    ```
    This array lists the class labels that the MobileNet SSD model can detect [2].

4.  **Load the Model:**
    ```
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    ```
    This loads the pre-trained MobileNet SSD model from the Caffe files [2].

5.  **Capture Video:**
    ```
    cap = cv2.VideoCapture(0)
    while True:
        _, image = cap.read()
        # ... object detection and display code ...
    ```
    This captures video frames from the default webcam in a loop [2].

6.  **Preprocess the Image:**
    ```
    height, width = image.shape, image.shape[1]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007, (300, 300), 130)
    net.setInput(blob)
    detected_objects = net.forward()
    ```
    This preprocesses each frame by resizing it to 300x300 pixels and converting it into a blob format suitable for the neural network [2].

7.  **Detect Objects:**
    ```
    for i in range(detected_objects.shape[2]):
        confidence = detected_objects[i][2]
        if confidence > min_confidence:
            # ... extract bounding box and class information ...
    ```
    This loop iterates through the detected objects, applying a confidence threshold to filter out low-probability detections, and extracts bounding box coordinates and class labels for the remaining objects [2].

8.  **Display Results:**
    ```
    cv2.rectangle(image, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y), colours[class_index], 3)
    cv2.putText(image, prediction_text, (upper_left_x, upper_left_y - 15 if upper_left_y > 30 else upper_left_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colours[class_index], 2)
    cv2.imshow("Detected Objects", image)
    ```
    This draws bounding boxes and class labels on the image for each detected object [2].

9.  **Exit Condition:**
    ```
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
    ```
    This allows the user to exit the program by pressing the 'q' key [1].

10. **Release Resources:**
    ```
    cap.release()
    cv2.destroyAllWindows()
    ```
    This releases the webcam and closes all OpenCV windows [2].

## Troubleshooting

If you encounter issues with the `MobileNetSSD.txt` file, follow these steps [1]:

1.  Delete the existing "MobileNetSSD.txt" file from the directory.
2.  Create a new "MobileNetSSD.txt" file in the same directory.
3.  Download the "MobileNetSSD.txt" file from the provided link [1].

## Additional Resources

-   **MobileNet SSD v2:**  An object detection model with 267 layers and 15 million parameters, providing real-time inference under compute constraints [3].
-   **Tutorial on MobileNet SSD with OpenCV:** [http://ebenezertechs.com/mobilenet-ssd-using-opencv-3-4-1-deep-learning-module-python/](http://ebenezertechs.com/mobilenet-ssd-using-opencv-3-4-1-deep-learning-module-python/) [4]
-   **Edge Impulse Tutorial:** [https://docs.edgeimpulse.com/docs/tutorials/end-to-end-tutorials/object-detection/object-detection](https://docs.edgeimpulse.com/docs/tutorials/end-to-end-tutorials/object-detection/object-detection) [5]

This project provides a foundation for real-time object detection using MobileNet SSD. Further improvements and customizations can be made to enhance its performance and applicability in various scenarios [6][7].
