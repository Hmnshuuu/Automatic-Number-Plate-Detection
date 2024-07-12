# Automatic Number Plate Detection

## Description

This project aims to detect vehicle number plates using YOLOv8 from Ultralytics, OpenCV, and PyTorch. The goal is to create an efficient and accurate system for automatic number plate recognition (ANPR), which can be used in various applications such as traffic management and security.

## Features

- **Accurate Detection**: Utilizes YOLOv8 for high-precision number plate detection.
- **Real-Time Processing**: Capable of processing video streams in real-time.
- **Easy Integration**: Modular design for seamless integration into other systems.
- **User-Friendly Interface**: Simple interface for running detection on images and videos.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Hmnshuuu/Automatic-Number-Plate-Detection.git
    ```

2. Navigate to the project directory:
    ```bash
    cd Automatic-Number-Plate-Detection
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Detect number plates in images:
    ```python
    import cv2
    from ultralytics import YOLO

    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Load the image
    image = cv2.imread('path/to/your/image.jpg')

    # Perform detection
    results = model(image)

    # Display the results
    results.show()
    ```

2. Detect number plates in video:
    ```python
    import cv2
    from ultralytics import YOLO

    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Open the video file
    cap = cv2.VideoCapture('path/to/your/video.mp4')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model(frame)

        # Display the results
        results.show()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    ```

## Project Structure

- `real_time_detection.py`: Script to run the application for real-time detection.
- `yolo_model.py`: Module for loading and utilizing the YOLOv8 model.
- `utils.py`: Utility functions for data processing and visualization.
- `data/`: Directory for storing input images and videos.

## Dataset

Prepare your dataset and place it in the `data/` directory. This dataset will be used to train and test the YOLOv8 model for accurate number plate detection.

## Training the Model

To train the YOLOv8 model on your dataset, follow these steps:

1. Prepare your dataset and place it in the `data/` directory.
2. Modify the training script as needed.
3. Run the training script:
    ```bash
    python train.py
    ```

4. The trained model will be saved in the specified directory.

## Contributing

Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- [Ultralytics](https://ultralytics.com/) for providing the YOLOv8 model.
- [OpenCV](https://opencv.org/) for computer vision tools.
- [PyTorch](https://pytorch.org/) for the deep learning framework.

---

Feel free to reach out if you have any questions or need further assistance. Happy coding!

[Himanshu]  
[himanshujangid364@gmail.com]  
[LinkedIn](https://www.linkedin.com/in/himanshuuu/)
