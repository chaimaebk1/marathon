
# Intelligent Photo Management for Sports Events

## Overview

This project aims to revolutionize the way photographers and participants interact with photos taken during sports events. By leveraging advanced machine learning and text mining techniques, this intelligent application offers a comprehensive solution for managing, organizing, and accessing event photos efficiently.

## Features

- **Bulk Photo Upload**: Allows photographers to quickly and securely upload large quantities of photos.
- **Automatic Photo Classification**: Utilizes image recognition models to identify and extract bib numbers from participants in photos, automatically organizing them accordingly.
- **Search by Bib Number**: Participants can easily find their photos by entering their bib number.
- **User-Friendly Interface**: Designed for both professional photographers and event participants, ensuring an intuitive and seamless experience.

## Screenshots
Annotation :
![image](https://github.com/YouEjj/marathon/assets/138532407/77816799-ea7e-44af-b402-4168af0e8eb6)
![image](https://github.com/YouEjj/marathon/assets/138532407/44e8366d-7dec-4d6d-9f37-6e08b9c02b81)
Python Interface:
![image](https://github.com/YouEjj/marathon/assets/138532407/2bd06bc2-76b5-486d-9dbb-1ba18b140ace)
![image](https://github.com/YouEjj/marathon/assets/138532407/42961b7c-77c1-470a-bd1f-315881c04ec1)


(Include your screenshots here)

## Getting Started

### Prerequisites

- Python 3.7 or higher
- PyTorch
- YOLOv8 model files
- Other Python libraries listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sports-event-photo-management.git
   cd sports-event-photo-management
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the YOLOv8 model weights and place them in the appropriate directory:
   ```bash
   mkdir -p models/yolov8
   # Download and place the yolov8s.pt file in the models/yolov8 directory
   ```

### Usage

1. Prepare your dataset and ensure it follows the structure defined in `data.yaml`.

2. Train the YOLOv8 model on your custom dataset:
   ```bash
   !yolo task=detect mode=train model=models/yolov8/yolov8s.pt data=data/data.yaml epochs=25 imgsz=800 plots=True
   ```

3. Evaluate the model's performance:
   - Review the training results in the generated plots.
   - Analyze the confusion matrix and performance metrics to ensure the model meets your requirements.

4. Use the trained model to classify and manage event photos:
   ```python
   from your_module import PhotoManager
   manager = PhotoManager(model_path="models/yolov8/yolov8s.pt", data_path="data/data.yaml")
   manager.classify_and_upload_photos("path_to_photos")
   ```

### Results Interpretation

- **Loss Graphs**: Show the decrease in box, classification, and DFL loss over epochs, indicating improved model performance.
- **Performance Metrics**: Precision, recall, and mAP metrics provide insight into the model's accuracy and effectiveness.
- **Confusion Matrix**: 
  ```
  Confusion Matrix:
  332  55
  14    0
  ```
  This matrix shows the model's ability to correctly identify objects (332 true positives) with some false positives (55) and false negatives (14).

- **Curves**: 
  - **Recall Curve**: Demonstrates the model's ability to detect most objects.
  - **Precision Curve**: Indicates the accuracy of the detected objects.
  - **F1 Score Curve**: Provides a balance between precision and recall.

### Future Work

- **Hyperparameter Tuning**: Further tuning of hyperparameters to enhance model performance.
- **Cross-Validation**: Implement cross-validation to ensure robust generalization.
- **Error Analysis**: Detailed analysis to identify and improve performance on difficult classes.

## Contributing

We welcome contributions to improve this project. Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

If you have any questions or suggestions, please feel free to reach out to us .

---
