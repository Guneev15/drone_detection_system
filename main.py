from ultralytics import YOLO

# Load a pre-trained model
model = YOLO('yolov8n.pt')

# Use the model
results = model.train(data='data.yaml', epochs=50)

# Save the model after training
model.save('yolov8n_trained.pt')