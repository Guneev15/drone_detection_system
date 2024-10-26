import os
from ultralytics import YOLO
import cv2
import numpy as np

def initialize_model():
    try:
        model = YOLO("C:\CODING\DroneDetctionSystem\DDT\yolov8n_trained.pt")
        print(f"Model loaded successfully")
        print(f"Model classes: {model.names}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def initialize_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return None
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera")
        return None
    
    print(f"Camera initialized successfully. Frame size: {frame.shape}")
    return cap

def draw_multiple_bounding_boxes(frame, x1, y1, x2, y2, score, color_main=(0, 255, 0)):
    """Draw multiple bounding boxes with different styles"""
    # Main solid box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color_main, 2)
    
    # Inner dashed box
    dash_length = 10
    gap_length = 5
    x, y = x1 + 5, y1 + 5
    w, h = x2 - x1 - 10, y2 - y1 - 10
    
    # Calculate points for dashed line
    pts = []
    for i in range(0, w, dash_length + gap_length):
        pts.extend([(x + i, y), (x + min(i + dash_length, w), y)])
        pts.extend([(x + i, y + h), (x + min(i + dash_length, w), y + h)])
    for i in range(0, h, dash_length + gap_length):
        pts.extend([(x, y + i), (x, y + min(i + dash_length, h))])
        pts.extend([(x + w, y + i), (x + w, y + min(i + dash_length, h))])
    
    # Draw dashed lines
    for i in range(0, len(pts), 2):
        if i + 1 < len(pts):
            cv2.line(frame, pts[i], pts[i+1], (255, 255, 255), 1)

def draw_confidence_bars(frame, x1, y1, x2, y2, score):
    """Draw confidence visualization bars"""
    bar_width = 100
    bar_height = 10
    bar_x = x1
    bar_y = y1 - 30
    
    # Draw background bar
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (0, 0, 0), -1)
    
    # Draw confidence bar
    conf_width = int(bar_width * score)
    conf_color = (0, int(255 * score), int(255 * (1 - score)))  # Color changes based on confidence
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + conf_width, bar_y + bar_height), conf_color, -1)
    
    # Draw border
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 1)

def main():
    model = initialize_model()
    if model is None:
        return

    cap = initialize_camera()
    if cap is None:
        return

    ret, frame = cap.read()
    H, W, _ = frame.shape

    video_path_out = 'webcam_output.mp4'
    out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), 30, (W, H))

    threshold = 0.5
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame")
            break

        try:
            results = model(frame, verbose=False)[0]
            
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}: Found {len(results.boxes)} detections")
            
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result

                if score > threshold:
                    # Convert coordinates to integers
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    
                    # Draw multiple bounding boxes
                    draw_multiple_bounding_boxes(frame, x1, y1, x2, y2, score)
                    
                    # Draw confidence bars
                    draw_confidence_bars(frame, x1, y1, x2, y2, score)
                    
                    # Prepare label with confidence and class name
                    class_name = model.names[int(class_id)]
                    label = f"{class_name}: {score*100:.2f}%"
                    
                    # Get text size
                    (label_width, label_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                    )
                    
                    # Draw label background
                    cv2.rectangle(
                        frame,
                        (x1, y1 - label_height - 45),  # Adjusted y position for confidence bar
                        (x1 + label_width, y1 - 35),
                        (0, 0, 0),
                        -1,
                    )
                    
                    # Draw text
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 40),  # Adjusted y position for confidence bar
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

        except Exception as e:
            print(f"Error during inference: {e}")
            continue

        # Show the frame
        cv2.imshow('Drone Detection', frame)
        out.write(frame)
        
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
