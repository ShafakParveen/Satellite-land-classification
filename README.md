# Satellite Land Use Classification using CNN

This project classifies satellite images into different land use / land cover categories using a Convolutional Neural Network (CNN).  
The model is trained on selected classes from the EuroSAT RGB dataset.

## Project Overview

Land Use and Land Cover (LULC) classification is an important task in remote sensing and geospatial analysis. It is widely used in:

- Urban planning
- Environmental monitoring
- Agriculture analysis
- Land resource management
- Satellite image interpretation

In this project, a CNN model is used to classify satellite images into three categories:

- Forest
- AnnualCrop
- Residential

The project includes image preprocessing, CNN model training, model evaluation, and result visualization.

## Dataset

This project uses the **EuroSAT RGB dataset**.

The dataset is not included in this repository because of its large size.  
To run the project, place the dataset in the following structure:

```bash
data/EuroSAT_data/
├── Forest/
├── AnnualCrop/
└── Residential/
```

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Pillow
- Scikit-learn

## Project Structure

```bash
Satellite-land-classification/
├── main.py
├── config.py
├── data_loader.py
├── model.py
├── train.py
├── evaluate.py
├── requirements.txt
├── README.md
├── outputs/
│   ├── accuracy_graph.png
│   ├── loss_graph.png
│   ├── confusion_matrix.png
│   ├── sample_predictions.png
│   └── classification_report.txt
└── data/
    └── EuroSAT_data/
        ├── Forest/
        ├── AnnualCrop/
        └── Residential/
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/ShafakParveen/Satellite-land-classification.git
```

### 2. Open the project folder

```bash
cd Satellite-land-classification
```

### 3. Install required libraries

```bash
pip install -r requirements.txt
```

### 4. Add the EuroSAT dataset

Place the dataset inside:

```bash
data/EuroSAT_data/
```

### 5. Run the project

```bash
python main.py
```

## Model Architecture

The CNN model consists of:

- Convolutional layers for feature extraction
- MaxPooling layers for reducing spatial dimensions
- Flatten layer for converting feature maps into a vector
- Dense layers for classification
- Dropout layer to reduce overfitting
- Softmax output layer for multi-class prediction

## Results

The model was trained on three EuroSAT classes:

- Forest
- AnnualCrop
- Residential

The project generates the following outputs:

- Accuracy graph
- Loss graph
- Confusion matrix
- Sample prediction images
- Classification report

## Output Visualizations

The following files are saved in the `outputs/` folder:

- `accuracy_graph.png`
- `loss_graph.png`
- `confusion_matrix.png`
- `sample_predictions.png`
- `classification_report.txt`

The classification report includes:

- Precision
- Recall
- F1-score
- Accuracy

## Limitations

- This project currently classifies only three EuroSAT classes.
- The complete EuroSAT dataset contains more land cover classes.
- This is a beginner-level implementation focused on understanding CNN-based satellite image classification.

## Future Work

Possible future improvements include:

- Classification of all 10 EuroSAT classes
- Use of Sentinel-2 or Landsat satellite imagery
- Integration with GIS / QGIS
- Area estimation of land cover classes
- Web application for image upload and prediction
- Comparison with machine learning models such as SVM, KNN, and Random Forest

## Author

**Shafak Parveen**  
B.Tech in Artificial Intelligence and Data Science  

Interested in Remote Sensing, GeoAI, and Satellite Image Analysis
