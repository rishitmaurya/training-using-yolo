import cv2
from ultralytics import YOLO

# Load the YOLO model (v8, as an example)
model = YOLO('runs\\detect\\train\\weights\\best.pt')  # Change this to your trained model

# Start webcam capture (0 for default webcam)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Resize the frame to 640x640
    resized_frame = cv2.resize(frame, (640, 640))

    # Run YOLO detection on the resized frame
    results = model(resized_frame)

    # Extract the first detection result
    predictions = results[0]

    # Get the bounding boxes (xyxy format), confidence scores, and class IDs
    boxes = predictions.boxes.xyxy.cpu().numpy()  # Bounding boxes in xyxy format
    confidences = predictions.boxes.conf.cpu().numpy()  # Confidence scores
    class_ids = predictions.boxes.cls.cpu().numpy()  # Class IDs

    # Load the class names (COCO dataset class names in this case)
    class_names = model.names  # The class names corresponding to the model

    # Draw bounding boxes on the frame
    for box, conf, class_id in zip(boxes, confidences, class_ids):
        # Extract the top-left and bottom-right coordinates
        x1, y1, x2, y2 = map(int, box)  # Convert to integer values

        # Calculate the center of the bounding box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Draw the rectangle
        cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw the center of the bounding box
        cv2.circle(resized_frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Red dot at the center

        # Add class name, confidence score, and center coordinates label
        label = f"{class_names[int(class_id)]} {conf:.2f}"
        cv2.putText(resized_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(resized_frame, f"Center: ({center_x}, {center_y})", (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the frame with detected objects, bounding boxes, and center coordinates
    cv2.imshow('YOLO Object Detection', resized_frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
