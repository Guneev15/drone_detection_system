## Table of Contents
- [Drone Detection System](#drone-detection-system)
- [Features](#features)
- [Dataset](#dataset)
- [Files](#files)
- [Example Outputs](#example-outputs)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)
- [License](#license)

# Drone Detection System

This is a Python-based drone detection system that utilizes YOLO (You Only Look Once) for object detection on a custom-trained drone dataset. It can detect drones in real time using a webcam or in video files. This project was created using OpenCV and YOLOv8.

## Features

- Real-time drone detection using a webcam
- Drone detection on pre-recorded video files
- Custom YOLOv8 model trained on a drone detection dataset

  ![napkin-selection (1)](https://github.com/user-attachments/assets/72bac339-df54-4872-8bc8-6932aa62ba70)


## Dataset

The model was trained on a custom drone detection dataset from [Kaggle's YOLO Drone Detection Dataset](https://www.kaggle.com/datasets/muki2003/yolo-drone-detection-dataset).

## Files

- `data.yaml`: Dataset configuration file for YOLOv8 training.
- `drone.mp4`: Example video file for testing the detection.
- `main.py`: Script for training the dataset.
- `predict_video.py`: Script for detection on a video file.
- `predict_webcam.py`: script for real-time detection using a webcam.
- `yolov8n_trained.pt`: Trained YOLOv8 model for drone detection.

![napkin-selection (2)](https://github.com/user-attachments/assets/0e6e0407-fddd-45aa-af96-e37283bc4e2a)

  
## Example Outputs
Here are some example outputs from the drone detection system:

### Webcam Detection Example
- **Description**: An example of drone detection using a webcam.
- **Output**:
  
![image](https://github.com/user-attachments/assets/b811b517-448b-4e8a-8e20-19db9e5de319)

![Screenshot 2024-10-26 131953](https://github.com/user-attachments/assets/c4f9aab5-5820-4314-b4da-400e622294d4)

### Video Detection Example
- **Description**: An example of drone detection on a pre-recorded video.
- **Output**:
https://github.com/user-attachments/assets/092f8b2b-c220-4f24-851d-97a08353f136

## Getting Started

### Prerequisites
Ensure that Python 3.12.7 is installed on your machine.

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/bhavishyasingla1/drone_detection.git
   cd drone_detection
   ```
2. **Install dependencies**:
      ```bash
      pip install -r requirements.txt
      ```

![napkin-selection (3)](https://github.com/user-attachments/assets/178d5a75-e9b8-441b-98b8-ae6893f52a9a)

      
## Usage

1. **Running Detection on Webcam**:

   Run the following command:
   ```bash
   python predict_webcam.py
   ```
   This will open your webcam and start detecting drones in real-time.

3. **Running Detection on a Video File**:

   Modify `video_path` in `predict_video.py` to your desired video file path:
   ```bash
   video_path = 'path/to/your/video.mp4'
   ```
  
   Run the script:
    ```bash
   python predict_video.py
   ```
   Output: Detected videos will be saved as `output.mp4` in the same directory as the script.

   ![napkin-selection (4)](https://github.com/user-attachments/assets/3a688d39-1440-443e-aa56-edb45037cb32)


## Training
To train the model on your dataset, ensure `data.yaml` is correctly configured, and then run `main.py` using the following command:

 ```bash
   python main.py
   ```
the output `yolov8n_trained.pt` will be saved in same directory as the script.

You can refer to the [Kaggle YOLO Drone Detection Dataset](https://www.kaggle.com/datasets/muki2003/yolo-drone-detection-dataset) for the dataset used. Watch [Computer Vision Engineer - YouTube Video](https://www.youtube.com/watch?v=m9fH9OWn8YM) to learn how to train your own custom dataset.

## Acknowledgments
- OpenCV for computer vision functionalities.
- YOLOv8 for object detection framework.
- Kaggle for providing the dataset.

![napkin-selection (5)](https://github.com/user-attachments/assets/8929079c-b638-49d3-9a7f-ac6c35e675b5)


## Contact
If you have any questions or feedback, feel free to reach out:
- Email: [bhavishyasingla2005@gmail.com](mailto:bhavishyasingla2005@gmail.com)
- GitHub: [Bhavishyasingla1](https://github.com/bhavishyasingla1)

## Troubleshooting
- **ModuleNotFoundError**: Ensure all dependencies are installed using `pip install -r requirements.txt`.
- **Performance issues**: Make sure your machine meets the required specifications for running YOLOv8.

  ## Future Improvements
- Implement multi-drone detection capabilities.
- Enhance model accuracy with more diverse datasets.
- Create a user interface for easier interaction.

  ![napkin-selection (6)](https://github.com/user-attachments/assets/41b79c2a-8d7d-488e-aa08-bd89195e90e4)


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.














   


