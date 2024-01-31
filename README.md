# Hand Gesture Recognition and Data Collection

## Introduction

This repository contains Python code for two related projects: a hand gesture data collection script and a hand gesture recognition script. The data collection script captures images of hand gestures using a webcam, while the recognition script utilizes a pre-trained model to classify the hand gestures in real-time.

### Features

- **Data Collection Script:**
  - Utilizes OpenCV and the cvzone library for hand detection.
  - Captures and saves images of hand gestures with predefined labels.
  - Allows users to customize the folder for saving images and set the number of images to capture.

- **Recognition Script:**
  - Combines hand detection using cvzone with a pre-trained gesture recognition model.
  - Recognizes and classifies hand gestures in real-time.
  - Displays bounding boxes and text annotations around the detected hands and their classifications.

## Requirements

- Python 3.x
- OpenCV
- cvzone library
- TensorFlow (for the recognition script)

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/hand-gesture-recognition.git
   ```

2. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the pre-trained gesture recognition model and labels from [Model.zip](link-to-model) and extract it into the `Model` folder.

## Usage

### Data Collection Script

1. Run the data collection script:

   ```bash
   python data_collection.py
   ```

2. Press the 's' key to save images. Images will be saved in the specified folder.

### Recognition Script

1. Run the recognition script:

   ```bash
   python recognition.py
   ```

2. The script will display real-time hand gesture recognition with bounding boxes and text annotations.

## Configuration

- **Folder and Counter:**
  - You can customize the folder for saving images in the data collection script by modifying the `folder` variable.
  - The `counter` variable keeps track of the number of captured images.

- **Model:**
  - The recognition script uses a pre-trained model located in the `Model` folder. Ensure the model and labels are correctly downloaded.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Hand detection powered by the [cvzone library](https://github.com/cvzone/cvzone).
- Gesture recognition model trained using TensorFlow.

## Author

[Nirmal Khadka]

Feel free to contribute, report issues, or suggest improvements. Happy coding!
