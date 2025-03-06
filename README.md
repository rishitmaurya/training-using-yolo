# YOLO Model Training Guide

This guide walks you through the steps required to train a YOLO (You Only Look Once) model for your custom dataset. Follow the instructions below to ensure you have all the necessary steps in place.

## Prerequisites
Before starting the training process, make sure you have:
- Python 3.x installed.
- YOLO framework set up (YOLOv5 or YOLOv4, depending on your preference).
- Necessary Python libraries (e.g., OpenCV, PyTorch, Ultralytics for YOLOv5).
- I have used **LABELME** for labeling the objects in the images, thus creating the `.json` files for that particular image for labels (used for validation).

## Steps for Training the YOLO Model

Make sure to change the path of the files according to your specific file locations.

### Step 1: Prepare Your Dataset
1. Create a **dataset folder**. Inside this folder, you should have two subfolders:
   - **images**: This folder contains all your image files.
   - **labels**: This folder contains all the corresponding `.txt` label files for each image.
   
   Example structure:
        dataset/
        ├── images/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        ├── labels/
        │   ├── image1.txt
        │   ├── image2.txt
        │   └── ...

### Step 2: Resize images and convert .json to .txt for labels (if necessary)
1. If you are training with an image size of **640 pixels**, and your images are larger, you need to resize them.
2. Use the provided **resize.py** file to resize your images to the desired size. You can adjust the parameters in the script to fit your needs.
3. If you have also used **LABELME** for labeling the images, that creates the `.json` files for labels, but YOLO uses `.txt` files for labels. Use the `convert_to_txt_yolo.py` file to convert those into useful formats.

### Step 3: Split Dataset into Training and Testing
1. After resizing the images, run the **test_train_split.py** script.
2. This script will split your dataset into training and testing datasets. It will save the split data into separate folders that will be used for training and validation.

### Step 4: Train the YOLO Model
1. Next, run the YOLO training command provided in the **yolo_command_for_training.txt** file.
   - Make sure to specify the correct path to your **data.yaml** file.
   - This file contains the paths to your training and validation datasets, along with the class names and the number of classes.

Example command for YOLOv5:
```bash
    yolo train data=path/to/your/data.yaml model=yolov8n.pt epochs=50 imgsz=640
```
### Step 5: Test the Model

Once the training process is completed, run the `test_model.py` script.

This will allow you to evaluate the trained model, check its performance, and ensure that everything is working correctly.

