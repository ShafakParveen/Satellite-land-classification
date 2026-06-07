# CNN-Based Satellite Image Classification for LULC Analysis

## About
This project uses a CNN model to classify EuroSAT satellite images into three land cover classes: Forest, AnnualCrop, and Residential.

## Dataset
Dataset used: EuroSAT

Classes:
- Forest
- AnnualCrop
- Residential

## Workflow
- Loaded satellite images from folders
- Resized images to 64 x 64 pixels
- Normalized pixel values
- Split dataset into training and testing data
- Built and trained a CNN model
- Evaluated the model using accuracy, loss, confusion matrix, classification report, and sample predictions

## How to Run

1. Clone the repository
2. Install requirements:
   pip install -r requirements.txt

3. Download EuroSAT dataset and place it inside:
   data/EuroSAT_data/

4. Run:
   python main.py

## Model
The CNN model uses Conv2D, MaxPooling2D, Flatten, Dense, Dropout, and Softmax layers.

## Results
The model achieved around 98-99% validation accuracy.

## Tools Used
Python, TensorFlow/Keras, NumPy, Matplotlib, Scikit-learn, Pillow

## Future Scope
This project can be improved by adding more classes, using Sentinel-2 or Landsat data, and performing area calculation or change detection.