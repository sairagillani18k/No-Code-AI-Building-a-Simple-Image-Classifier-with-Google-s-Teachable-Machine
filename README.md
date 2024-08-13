# No-Code-AI-Building-a-Simple-Image-Classifier-with-Google-s-Teachable-Machine
# Teachable Machine Image Classification Project

This project demonstrates how to create an image classification model using Google's Teachable Machine and deploy it in a Python environment. The model was trained to distinguish between images of cats and dogs using a simple, user-friendly interface. This README file provides a step-by-step guide to replicating the project, along with Python code to run the model on a video feed or images.

## Project Overview

The Teachable Machine is an easy-to-use tool for creating machine learning models without any coding experience. In this project, we trained an image classification model and exported it to a TensorFlow SavedModel format. The model is then used in a Python environment to classify video frames.

## Steps to Create and Use the Model

### 1. Start a New Project

- Visit the [Teachable Machine](https://teachablemachine.withgoogle.com/train) website.
- Select the `Image Project` option.
- You can either start a new project or open an existing one from your Google Drive.

![New Project](/Pictures/11.png)

### 2. Choose Your Model Type

- You have the option to choose between a `Standard image model` and an `Embedded image model`. For this project, we used the `Standard image model`.

![Model Type](/Pictures/2.png)

### 3. Train Your Model

- Add image samples for each class (e.g., Cats and Dogs).
- Train the model by clicking the `Train Model` button. The model will learn to differentiate between the provided image classes.

![Training the Model](/Pictures/3.png)

### 4. Export the Model

- Once the training is complete, click on `Export Model`.
- Select the `TensorFlow SavedModel` format to export your trained model.
- Download the model and save it to your local machine.

![Exporting the Model](/Pictures/5.png)

### 5. Running the Model in Python

- Install the required libraries:

  ```bash
  pip install tensorflow opencv-python numpy
  
- Enter the correct paths for the model files, labels.txt, and the video on which you want to perform inference.

- Enter the following command to run the script
```bash
  python3 main.py
