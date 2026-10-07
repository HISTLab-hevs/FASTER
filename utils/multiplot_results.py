"""Command-line tool that plots accuracy across multiple experiments."""

import argparse
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd


def parse_arguments():
    """Parse command-line arguments for multi-experiment plotting."""
    parser = argparse.ArgumentParser(description="Plot accuracy results for multiple experiments.")
    parser.add_argument('folders', type=str, nargs='+', help="Paths to the folders containing result CSV files.")
    return parser.parse_args()


def load_data(folder):
    """Load CSV summaries for one experiment root."""
    clients_file = os.path.join(folder, 'clients_results.csv')
    validation_file = os.path.join(folder, 'validation_results.csv')
    test_file = os.path.join(folder, 'test_results.csv')

    clients_data = pd.read_csv(clients_file, index_col="Epoch")
    validation_data = pd.read_csv(validation_file, index_col="Epoch")
    test_data = pd.read_csv(test_file, index_col="Epoch") if os.path.exists(test_file) else None

    return clients_data, validation_data, test_data


def generate_color_palette():
    """Generate paired colors for client bands and aggregate markers."""
    base_colors = list(mcolors.TABLEAU_COLORS.values())

    color_pairs = []
    for color in base_colors:
        lighter_color = mcolors.to_rgba(color, alpha=0.5)
        darker_color = mcolors.to_rgba(color)
        color_pairs.append((lighter_color, darker_color))

    return color_pairs


def plot_results(folders):
    """Plot client (train) and validation/test accuracy curves for multiple experiments."""
    plt.figure(figsize=(12, 8))

    color_palette = generate_color_palette()

    if len(folders) > len(color_palette):
        raise ValueError("Number of experiments exceeds available fixed colors in the palette.")

    max_epochs = 0

    for i, folder in enumerate(folders):
        experiment_name = os.path.basename(os.path.normpath(folder))

        clients_data, validation_data, test_data = load_data(folder)

        client_avg = clients_data['Overall_Avg_Clients_Accuracy']
        client_std = clients_data['Overall_Std_Clients_Accuracy']
        validation_avg = validation_data['Overall_Avg_Validation_Accuracy']
        validation_std = validation_data['Overall_Std_Validation_Accuracy']

        total_epochs = max(len(clients_data), len(validation_data))
        max_epochs = max(max_epochs, total_epochs)

        client_color, aggregate_color = color_palette[i]

        plt.plot(client_avg.index, client_avg, label=f'{experiment_name} - Clients Accuracy',
                 color=client_color, linewidth=2.5)
        plt.fill_between(client_avg.index, client_avg - client_std, client_avg + client_std,
                         color=client_color, alpha=0.4)

        plt.plot(validation_avg.index, validation_avg, label=f'{experiment_name} - Validation Accuracy',
                 color=aggregate_color, linewidth=2.5)
        plt.fill_between(validation_avg.index, validation_avg - validation_std, validation_avg + validation_std,
                         color=aggregate_color, alpha=0.25)

        if test_data is not None and 'Overall_Avg_Test_Accuracy' in test_data.columns:
            test_avg = test_data['Overall_Avg_Test_Accuracy']
            test_std = test_data['Overall_Std_Test_Accuracy']
            plt.plot(test_avg.index, test_avg, label=f'{experiment_name} - Test Accuracy',
                     color=aggregate_color, linewidth=1.5, linestyle='--')
            plt.fill_between(test_avg.index, test_avg - test_std, test_avg + test_std,
                             color=aggregate_color, alpha=0.15)

    plt.xticks(range(1, max_epochs + 2), fontsize=24)
    plt.xlim(1, max_epochs + 1)
    plt.yticks(fontsize=24)

    plt.xlabel("Epochs", fontsize=24)
    plt.ylabel("Accuracy", fontsize=24)
    plt.legend(fontsize=20)
    plt.grid(True)

    plt.tight_layout()

    experiment_names = [os.path.basename(os.path.normpath(folder)) for folder in folders]
    file_name = f"comparison_{'_'.join(experiment_names)}.svg"
    plt.savefig(file_name, format='svg', bbox_inches='tight')
    print(f"Plot generated and saved as '{file_name}'.")

def main():
    """Parse arguments and generate the multi-experiment comparison plot."""
    args = parse_arguments()
    plot_results(args.folders)


if __name__ == "__main__":
    main()
