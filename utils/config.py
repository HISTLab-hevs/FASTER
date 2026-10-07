"""Loading, dumping, and alias normalization for YAML run configuration files."""

import yaml


SERVER_WARMUP_EPOCHS_KEY = "server_warmup_epochs"
LEGACY_SERVER_WARMUP_EPOCHS_KEY = "global_model_epochs"


def load_config(filepath='config.yaml'):
    """Load configuration values from a YAML file."""
    with open(filepath, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            config = None
    return config


def dump_config(dest_path, config):
    """Persist configuration values to a YAML file."""
    with open(dest_path, 'w') as f:
        try:
            yaml.safe_dump(config, f, default_flow_style=False)
            print(f"Configuration saved in {dest_path}")
        except yaml.YAMLError as exc:
            print(f"Error while saving configuration: {exc}")


def normalize_run_config_aliases(config):
    """Normalize legacy run-config aliases to their canonical names."""
    if not isinstance(config, dict):
        return config

    if SERVER_WARMUP_EPOCHS_KEY not in config and LEGACY_SERVER_WARMUP_EPOCHS_KEY in config:
        config[SERVER_WARMUP_EPOCHS_KEY] = config[LEGACY_SERVER_WARMUP_EPOCHS_KEY]

    if LEGACY_SERVER_WARMUP_EPOCHS_KEY in config:
        config.pop(LEGACY_SERVER_WARMUP_EPOCHS_KEY, None)

    return config
