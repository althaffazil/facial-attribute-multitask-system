from configs.config import Config
from data.dataset import get_dataloader
from models.multitask_model import MultiTaskModel
from utils.trainer import train_model

def main():
    config = Config()

    train_loader = get_dataloader("train", config)
    val_loader = get_dataloader("validation", config)

    model = MultiTaskModel().to(config.DEVICE)

    train_model(model, train_loader, val_loader, config)

if __name__ == "__main__":
    main()
