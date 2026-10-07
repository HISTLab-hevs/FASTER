"""Command-line tool that plots client, validation, and test accuracy for one experiment."""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def parse_arguments():
    """Parse command-line arguments for single-experiment plotting."""
    parser = argparse.ArgumentParser(description="Plot client, validation, and test accuracy results.")
    parser.add_argument('root_folder', type=str, help="Path to the folder containing result CSV files.")
    return parser.parse_args()


def load_data(root_folder):
    """Load the CSV summaries for one results root."""
    clients_file = os.path.join(root_folder, 'clients_results.csv')
    validation_file = os.path.join(root_folder, 'validation_results.csv')
    test_file = os.path.join(root_folder, 'test_results.csv')

    clients_data = pd.read_csv(clients_file, index_col="Epoch")
    validation_data = pd.read_csv(validation_file, index_col="Epoch")
    test_data = pd.read_csv(test_file, index_col="Epoch") if os.path.exists(test_file) else None

    return clients_data, validation_data, test_data


def plot_results(clients_data, validation_data, root_folder, test_data=None):
    """Plot client (train) and validation/test accuracy aggregates for one experiment root."""
    client_avg = clients_data['Overall_Avg_Clients_Accuracy']
    client_std = clients_data['Overall_Std_Clients_Accuracy']
    validation_avg = validation_data['Overall_Avg_Validation_Accuracy']
    validation_std = validation_data['Overall_Std_Validation_Accuracy']

    plt.figure(figsize=(10, 6))

    plt.plot(client_avg.index, client_avg, label='Client Avg Accuracy', color='blue')
    plt.fill_between(client_avg.index, client_avg - client_std, client_avg + client_std, color='blue', alpha=0.3)

    plt.plot(validation_avg.index, validation_avg, label='Validation Avg Accuracy', color='green', marker='o')
    plt.fill_between(validation_avg.index, validation_avg - validation_std, validation_avg + validation_std, color='green', alpha=0.2)

    if test_data is not None and 'Overall_Avg_Test_Accuracy' in test_data.columns:
        test_avg = test_data['Overall_Avg_Test_Accuracy']
        test_std = test_data['Overall_Std_Test_Accuracy']
        plt.plot(test_avg.index, test_avg, label='Test Avg Accuracy', color='orange', marker='o')
        plt.fill_between(test_avg.index, test_avg - test_std, test_avg + test_std, color='orange', alpha=0.2)

    total_epochs = max(len(clients_data), len(validation_data), len(test_data) if test_data is not None else 0)
    plt.xticks(range(1, total_epochs + 1))

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Mean and Standard Deviation of Accuracy: Clients vs Validation/Test")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(root_folder, 'accuracy_plot.png'))
    plt.show()
    print("Plot generated and saved as 'accuracy_plot.png'.")


def main():
    """Parse arguments and generate the experiment plot."""
    args = parse_arguments()
    clients_data, validation_data, test_data = load_data(args.root_folder)
    plot_results(clients_data, validation_data, args.root_folder, test_data=test_data)


if __name__ == "__main__":
    main()
