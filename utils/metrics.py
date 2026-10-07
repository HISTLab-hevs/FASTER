"""Formatting, serialization, storage, and plotting of run metrics."""

import argparse
import json
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker


def print_metrics(metrics):
    """Print per-client train metrics plus validation and test metrics in a readable format.

    Per-client metrics are evaluated on each client's own training data; the
    validation/test sections describe the aggregated global model.
    """

    def format_metric(value, format_str):
        return format_str.format(value) if value is not None else "N/A"

    print("\nClient Metrics (evaluated on client training data):")
    for client_idx, client_metrics in enumerate(metrics["clients"]):
        print(f"Client {client_idx + 1}:")
        for idx, (loss, acc, precision, recall, f1, roc_auc) in enumerate(
            zip(
                client_metrics["loss"],
                client_metrics["accuracy"],
                client_metrics["precision"],
                client_metrics["recall"],
                client_metrics["f1"],
                client_metrics["roc_auc"],
            )
        ):
            print(
                f"{idx}: Loss={format_metric(loss, '{:.4f}')}, Accuracy={format_metric(acc, '{:.4f}')}, "
                f"Precision={format_metric(precision, '{:.4f}')}, Recall={format_metric(recall, '{:.4f}')}, "
                f"F1={format_metric(f1, '{:.4f}')}, ROC AUC={format_metric(roc_auc, '{:.4f}')}"
            )

        print()

    print("\nValidation Metrics:")
    for idx, (loss, acc, precision, recall, f1, roc_auc) in enumerate(
        zip(
            metrics["aggregated_val"]["loss"],
            metrics["aggregated_val"]["accuracy"],
            metrics["aggregated_val"]["precision"],
            metrics["aggregated_val"]["recall"],
            metrics["aggregated_val"]["f1"],
            metrics["aggregated_val"]["roc_auc"],
        )
    ):
        print(
            f"{idx}: Loss={format_metric(loss, '{:.4f}')}, Accuracy={format_metric(acc, '{:.4f}')}, "
            f"Precision={format_metric(precision, '{:.4f}')}, Recall={format_metric(recall, '{:.4f}')}, "
            f"F1={format_metric(f1, '{:.4f}')}, ROC AUC={format_metric(roc_auc, '{:.4f}')}"
        )

    if metrics.get("aggregated_test", {}).get("accuracy"):
        print("\nTest Metrics:")
        for idx, (loss, acc, precision, recall, f1, roc_auc) in enumerate(
            zip(
                metrics["aggregated_test"]["loss"],
                metrics["aggregated_test"]["accuracy"],
                metrics["aggregated_test"]["precision"],
                metrics["aggregated_test"]["recall"],
                metrics["aggregated_test"]["f1"],
                metrics["aggregated_test"]["roc_auc"],
            )
        ):
            print(
                f"{idx}: Loss={format_metric(loss, '{:.4f}')}, Accuracy={format_metric(acc, '{:.4f}')}, "
                f"Precision={format_metric(precision, '{:.4f}')}, Recall={format_metric(recall, '{:.4f}')}, "
                f"F1={format_metric(f1, '{:.4f}')}, ROC AUC={format_metric(roc_auc, '{:.4f}')}"
            )


def convert_to_serializable(obj):
    """Convert unsupported JSON values such as NumPy arrays into serializable data."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def store_metrics(filepath, metrics):
    """Serialize metrics to JSON, overwriting any existing file."""
    serializable_metrics = json.loads(
        json.dumps(metrics, default=convert_to_serializable)
    )

    with open(filepath, "w") as f:
        json.dump(serializable_metrics, f, indent=4)

    print(f"Metrics stored and file {filepath} overwritten.")


def plot_metrics(
    metrics,
    show=False,
    save=None,
    include_validation=True,
    include_test=True,
    include_clients=True,
    losses=False,
):
    """Plot validation, test, and client loss/accuracy curves over epochs."""
    validation_epochs = list(range(1, len(metrics["aggregated_val"]["loss"]) + 1))
    test_epochs = list(range(1, len(metrics.get("aggregated_test", {}).get("loss", [])) + 1))

    colors = plt.cm.get_cmap("tab10", len(metrics["clients"]) + 2)

    if losses:
        color_idx = 0

        plt.figure(figsize=(12, 6))

        if include_validation:
            plt.plot(
                validation_epochs,
                metrics["aggregated_val"]["loss"],
                label="Validation",
                marker="o",
                color=colors(color_idx),
            )
            color_idx += 1

        if include_test and metrics.get("aggregated_test", {}).get("loss"):
            plt.plot(
                test_epochs,
                metrics["aggregated_test"]["loss"],
                label="Test",
                marker="o",
                color=colors(color_idx),
            )
            color_idx += 1

        if include_clients:
            for idx, client in enumerate(metrics["clients"]):
                client_epochs = list(range(1, len(client["loss"]) + 1))
                plt.plot(
                    client_epochs,
                    client["loss"],
                    label=f"Client {idx + 1}",
                    color=colors(color_idx),
                )
                color_idx = (
                    color_idx + 1
                ) % colors.N

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Model Loss Over Epochs")
        plt.legend()
        plt.grid(True)

        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))

        if save:
            plt.savefig(save)
        if show:
            plt.show()
        plt.close()

    color_idx = 0

    plt.figure(figsize=(12, 6))

    if include_validation:
        plt.plot(
            validation_epochs,
            metrics["aggregated_val"]["accuracy"],
            label="Validation",
            marker="o",
            color=colors(color_idx),
        )
        color_idx += 1

    if include_test and metrics.get("aggregated_test", {}).get("accuracy"):
        plt.plot(
            test_epochs,
            metrics["aggregated_test"]["accuracy"],
            label="Test",
            marker="o",
            color=colors(color_idx),
        )
        color_idx += 1

    if include_clients:
        for idx, client in enumerate(metrics["clients"]):
            client_epochs = list(range(1, len(client["accuracy"]) + 1))
            plt.plot(
                client_epochs,
                client["accuracy"],
                label=f"Client {idx + 1}",
                color=colors(color_idx),
            )
            color_idx = (color_idx + 1) % colors.N

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tick_params(axis="y", labelsize=16)
    plt.tick_params(axis="x", labelsize=16)
    plt.title("Model Accuracy Over Epochs")
    plt.legend()
    plt.grid(True)

    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))

    if save:
        plt.savefig(save)
    if show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Metrics")
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Folder containing the metrics.json file",
    )
    args = parser.parse_args()

    folder = args.folder
    with open(os.path.join(folder, "metrics.json"), "r") as f:
        metrics = json.load(f)

    plot_metrics(
        metrics,
        show=True,
        save=os.path.join(folder, "metrics_validation.png"),
        include_validation=True,
        include_test=False,
        include_clients=False,
    )
    plot_metrics(
        metrics,
        show=True,
        save=os.path.join(folder, "metrics_clients.png"),
        include_validation=False,
        include_test=False,
        include_clients=True,
    )
    plot_metrics(
        metrics,
        show=True,
        save=os.path.join(folder, "metrics_test.png"),
        include_validation=False,
        include_test=True,
        include_clients=False,
    )
