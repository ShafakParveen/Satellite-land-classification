DATASET_PATH = "data/EuroSAT_data"

CLASSES = [
    "Forest",
    "AnnualCrop",
    "Residential",
    ]
IMAGE_SIZE = 64
EPOCHS = 9
BATCH_SIZE = 32
TEST_SIZE = 0.30
RANDOM_STATE = 42
MODEL_SAVE_PATH = "outputs/cnn_lulc_eurosat_model.keras"