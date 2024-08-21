import os
import json

def parse_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

# Example usage
if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)  # Get the directory of the script
    config_file_path = os.path.join(script_dir, 'config.json')  # Build the full path
    config = parse_config(config_file_path)
    print(config)
