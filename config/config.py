import os

import os

def load_config(config_file):
    config_file = os.path.join(os.path.dirname(__file__), config_file)
    with open(config_file, "r") as file:
        import yaml
        config = yaml.safe_load(file)
    return config


