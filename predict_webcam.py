import os
from ultralytics import YOLO
import cv2

# Load the trained model
print("Loading the trained model...")
model = YOLO('yolov8n_trained.pt')

# Open the webcam
print("Opening the webcam...")
cap = cv2.VideoCapture(0)  # 0 is the default webcam

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_path_out = 'output_webcam.mp4'
out = cv2.VideoWriter(video_path_out, fourcc, fps, (width, height))

print("Starting webcam processing...")
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    print(f"Processing frame {frame_count}...")

    # Perform inference on the frame
    results = model(frame)

    # Check if any objects were detected
    if len(results) == 0:
        print(f"No objects detected in frame {frame_count}")
        continue

    # Iterate over the results and draw bounding boxes
    for result in results:
        print(f"Detected {len(result.boxes)} objects in frame {frame_count}")
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = box.conf[0]
            
            # Check if class_id is in result.names
            if class_id in result.names:
                label = result.names[class_id]
            else:
                label = f'Class {class_id}'

            print(f"Drawing box for {label} with confidence {confidence:.2f} in frame {frame_count}")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow('Webcam', frame)

    # Write the frame to the output video
    out.write(frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Releasing resources...")

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved to {video_path_out}")