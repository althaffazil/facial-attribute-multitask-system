from sklearn.metrics import accuracy_score, f1_score
import torch


def compute_metrics(outputs, targets):
    probs = torch.sigmoid(outputs)
    preds = (probs > 0.5).int()
    targets = targets.int()

    metrics = {}

    for i, label in enumerate(["Gender", "Smile"]):
        acc = accuracy_score(targets[:, i], preds[:, i])
        f1 = f1_score(targets[:, i], preds[:, i])
        metrics[label] = {"Accuracy": acc, "F1": f1}

    return metrics
