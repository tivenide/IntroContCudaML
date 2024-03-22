import argparse
import json

from utils import load_data
import training

def main(config):
    if config.get("data_download")["download_data"]:
        # DOWNLOAD
        print("Download data")
        load_data(path_input=config["data_download"]["path"], download=True)

    if config.get("training")["train_on_data"]:
        # TRAINING
        print("Train on data")
        try:
            training.run(
                config.get("training")
            )
        except KeyboardInterrupt:
            print("Training interrupted by user.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model with a given configuration file."
    )
    parser.add_argument(
        "-c", "--config_path", type=str, help="The path to the configuration file."
    )
    args = parser.parse_args()

    config_path = args.config_path

    assert config_path.endswith(".json"), "Configuration file must be a JSON file."
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            config["config_name"] = config_path.split("/")[-1].split(".")[0]
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Configuration file not found at {config_path}") from e
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            "Configuration file is not a valid JSON file.", e.doc, e.pos
        ) from e

    main(config)