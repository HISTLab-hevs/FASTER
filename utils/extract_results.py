"""Command-line tool that extracts run metrics into CSV summaries."""

import os
import json
import pandas as pd
import argparse


SCOPES = {
    "validation": "aggregated_val",
    "test": "aggregated_test",
}


def parse_arguments():
    """Parse command-line arguments for CSV extraction utilities."""
    parser = argparse.ArgumentParser(description="Process experiment results and save to structured CSV files.")
    parser.add_argument('root_folder', type=str, help="Path to the root folder containing experiment folders.")
    return parser.parse_args()


def load_metrics(metrics_file):
    """Load one experiment metrics file from JSON."""
    with open(metrics_file, 'r') as f:
        return json.load(f)


def process_metrics(metrics, experiment_name, key):
    """Convert one experiment section into a tabular accuracy DataFrame.

    Per-client accuracy is evaluated on each client's training data; validation/test
    sections describe the aggregated global model.
    """
    data = {}
    bucket_key = SCOPES.get(key, key)
    num_epochs = len(metrics[bucket_key][0]['accuracy']) if key == "clients" else len(metrics[bucket_key]['accuracy'])

    if key == "clients":
        num_clients = len(metrics['clients'])
        for client_id in range(num_clients):
            client_accuracies = metrics['clients'][client_id]['accuracy']
            data[f"{experiment_name}_Client{client_id + 1}_Accuracy"] = client_accuracies

    else:
        title = key.capitalize()
        data[f"{experiment_name}_{title}_Accuracy"] = metrics[bucket_key]['accuracy']

    return pd.DataFrame(data, index=range(1, num_epochs + 1))


def process_all_experiments(root_folder, key):
    """Combine one metric section across all experiment folders."""
    combined_data = pd.DataFrame()

    for experiment_name in os.listdir(root_folder):
        experiment_path = os.path.join(root_folder, experiment_name)
        metrics_file = os.path.join(experiment_path, 'metrics.json')

        if os.path.isfile(metrics_file):
            metrics = load_metrics(metrics_file)
            experiment_data = process_metrics(metrics, experiment_name, key)
            combined_data = pd.concat([combined_data, experiment_data], axis=1)

    combined_data.index.name = "Epoch"

    accuracy_columns = [col for col in combined_data.columns if '_Accuracy' in col]
    label = key.capitalize()
    combined_data[f'Overall_Avg_{label}_Accuracy'] = combined_data[accuracy_columns].mean(axis=1)
    combined_data[f'Overall_Std_{label}_Accuracy'] = combined_data[accuracy_columns].std(axis=1)

    return combined_data


def save_to_csv(data, root_folder, filename):
    """Write a processed results DataFrame to CSV."""
    output_file = os.path.join(root_folder, filename)
    data.to_csv(output_file, index_label="Epoch")
    print(f"CSV file successfully generated: {output_file}")


def main():
    """Generate client, validation, and test CSV summaries for one results root."""
    args = parse_arguments()

    """Export per-client accuracy summaries plus overall statistics."""
    clients_data = process_all_experiments(args.root_folder, 'clients')
    save_to_csv(clients_data, args.root_folder, 'clients_results.csv')

    """Export validation accuracy summaries."""
    validation_data = process_all_experiments(args.root_folder, 'validation')
    save_to_csv(validation_data, args.root_folder, 'validation_results.csv')

    """Export test accuracy summaries."""
    test_data = process_all_experiments(args.root_folder, 'test')
    save_to_csv(test_data, args.root_folder, 'test_results.csv')


if __name__ == "__main__":
    main()
