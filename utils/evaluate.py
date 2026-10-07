"""Model evaluation returning loss plus classification metrics."""

import torch
from torch import nn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import torch.nn.functional as F
import numpy as np


def evaluate(model, test_loader, device="cpu"):
    """Evaluate a model on a dataloader and return loss plus classification metrics."""

    model.to(device)
    model.eval()
    if test_loader is None or len(getattr(test_loader, 'dataset', [])) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, None
    total_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []
    all_probs = []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
            targets = target.view(-1).long()
            loss = criterion(output, targets)

            total_loss += loss.item() * targets.size(0)

            probs = F.softmax(output, dim=1)
            _, predicted = torch.max(output.data, 1)

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    ds_len = len(test_loader.dataset)
    average_loss = total_loss / ds_len if ds_len else 0.0
    # Accuracy is reported as a 0-1 fraction so it shares the same scale as
    # precision/recall/f1/roc_auc (all sklearn macro scores). Reporting it as a
    # 0-100 percentage made it inconsistent with the other metric cards/charts.
    accuracy = (correct / total) if total else 0.0

    if not all_targets:
        return average_loss, accuracy, 0.0, 0.0, 0.0, None

    precision = precision_score(all_targets, all_predictions, average="macro", zero_division=0)
    recall = recall_score(all_targets, all_predictions, average="macro", zero_division=0)
    f1 = f1_score(all_targets, all_predictions, average="macro", zero_division=0)
    roc_auc = None
    if all_probs:
        try:
            y_true = np.asarray(all_targets)
            y_score = np.asarray(all_probs)
            n_score_classes = y_score.shape[1] if y_score.ndim == 2 else 0
            labels = list(range(n_score_classes))
            unique_targets = np.unique(y_true)

            if n_score_classes == 2:
                # Binary ROC AUC is only defined when both classes are present.
                if unique_targets.size >= 2:
                    roc_auc = roc_auc_score(y_true, y_score[:, 1])
            elif n_score_classes > 2:
                # Pass explicit labels so missing classes do not break multiclass ROC AUC.
                if unique_targets.size >= 2:
                    roc_auc = roc_auc_score(
                        y_true,
                        y_score,
                        multi_class="ovr",
                        labels=labels,
                    )
        except ValueError:
            roc_auc = None

    return average_loss, accuracy, precision, recall, f1, roc_auc
