"""Local client training loop shared by the federated aggregation methods."""

from torch import nn

from utils.evaluate import evaluate

def train(
    model,
    optimizer,
    train_loader,
    val_loader=None,
    epochs=1,
    device="cpu",
    global_params=None,
    mu=0,
    eval_split_name="Validation",
    progress_callback=None,
):
    """Train a model and collect per-epoch training and validation metrics."""

    model.to(device)
    metrics = {
        "loss": [],
        "val_loss": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "roc_auc": [],
    }
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            targets = target.view(-1).long()
            loss = criterion(output, targets)

            if global_params is not None and mu > 0:
                proximal_term = 0.0
                for param, global_param in zip(model.parameters(), global_params):
                    proximal_term += ((param - global_param.detach()) ** 2).sum()
                loss += (mu / 2) * proximal_term

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)
            total += target.size(0)
        running_loss = (total_loss / total) if total else 0.0

        # The eval loader is optional: the server warm-up passes the validation set and
        # clients pass their own training set, so the recorded metrics describe whichever
        # split is supplied (eval_split_name labels it in the log). When supplied, all
        # metrics (loss included) come from evaluate() on that split so they are coherent
        # and free of the running/proximal-term effects of the in-epoch loss.
        if val_loader:
            val_loss, accuracy, precision, recall, f1, roc_auc = evaluate(model, val_loader, device)
            print(
                f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss}, {eval_split_name} Loss: {val_loss}, {eval_split_name} Accuracy: {accuracy}, {eval_split_name} Precision: {precision}, {eval_split_name} Recall: {recall}, F1: {f1}, ROC AUC: {roc_auc}"
            )
            metrics["loss"].append(val_loss)
            metrics["val_loss"].append(val_loss)
            metrics["accuracy"].append(accuracy)
            metrics["precision"].append(precision)
            metrics["recall"].append(recall)
            metrics["f1"].append(f1)
            metrics["roc_auc"].append(roc_auc)
        else:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss}")
            metrics["loss"].append(running_loss)

        if callable(progress_callback):
            try:
                progress_callback(epoch + 1, epochs)
            except Exception:
                pass
    return metrics
