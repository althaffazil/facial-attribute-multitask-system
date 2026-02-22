import torch
import torch.nn as nn
from tqdm import tqdm
from utils.metrics import compute_metrics

def train_model(model, train_loader, val_loader, config):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    scaler = torch.cuda.amp.GradScaler(enabled=config.MIXED_PRECISION)

    best_val_loss = float("inf")

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        train_loss = 0

        for images, targets in tqdm(train_loader):
            images = images.to(config.DEVICE)
            targets = targets.to(config.DEVICE)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=config.MIXED_PRECISION):
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        print(f"Epoch {epoch+1} Train Loss: {train_loss/len(train_loader):.4f}")

        validate(model, val_loader, config, criterion)

    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)


def validate(model, loader, config, criterion):
    model.eval()
    val_loss = 0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(config.DEVICE)
            targets = targets.to(config.DEVICE)

            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())

    outputs = torch.cat(all_outputs)
    targets = torch.cat(all_targets)

    metrics = compute_metrics(outputs, targets)

    print(f"Validation Loss: {val_loss/len(loader):.4f}")
    print(metrics)
