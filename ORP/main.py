import numpy as np
import cv2

image_path = 'models/roompeople.jpg'
prototxt_path = 'models/MobileNetSSD_deploy.prototxt.txt'
model_path = 'models/MobileNetSSD_deploy.caffemodel'
min_confidence = 0.2

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



np.random.seed(543210)
colours = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

cap = cv2.VideoCapture(0)

while True:

    _, image = cap.read()

    height, width = image.shape[0], image.shape[1]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007, (300, 300), 130)

    net.setInput(blob)
    detected_objects = net.forward()

    print(detected_objects[0][0][8])

    for i in range(detected_objects.shape[2]):

        confidence = detected_objects[0][0][i][2]

        if confidence > min_confidence:
            class_index = int(detected_objects[0, 0, i, 1])

            upper_left_x = int(detected_objects[0, 0, i, 3] * width)
            upper_left_y = int(detected_objects[0, 0, i, 4] * height)
            lower_right_x = int(detected_objects[0, 0, i, 5] * width)
            lower_right_y = int(detected_objects[0, 0, i, 6] * height)

            prediction_text = f"{classes[class_index]}: {confidence:.2f}%"
            cv2.rectangle(image, (upper_left_x, upper_left_y, lower_right_x, lower_right_y), colours[class_index], 3)
            cv2.putText(image, prediction_text,
                        (upper_left_x, upper_left_y - 15 if upper_left_y > 30 else upper_left_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colours[class_index], 2)

    cv2.imshow("Detected Objects", image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
