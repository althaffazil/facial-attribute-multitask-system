import torch
class Config:
    DATASET_NAME = "abhiyanta/celebA-gender-smile"
    IMAGE_SIZE = 128
    BATCH_SIZE = 32
    NUM_EPOCHS = 5
    LR = 1e-4
    NUM_WORKERS = 2

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MIXED_PRECISION = torch.cuda.is_available()

    MODEL_SAVE_PATH = "models/best_model.pth"
