import argparse

from omegaconf import OmegaConf
from train_k_fold import train_k_fold

def main(argv=None):
    """
    Main entry point for training.
    Selects the training pipeline based on the configuration.

        python run.py [--config configs/train_config.yaml] [data.parquet_path=... logging.run_name=... ...]

    Trailing ``key=value`` arguments override entries of the config file (OmegaConf dotlist syntax).
    """
    parser = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/train_config.yaml")
    args, overrides = parser.parse_known_args(argv)
    config = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(overrides))

    if config.pipeline == "lightning":
        print("Starting Lightning pipeline...")
        # The parameters of the selected model
        config.model.params = config.model[config.model.name]
        train_k_fold(config)
    else:
        raise ValueError(f"Unknown pipeline: {config.pipeline}")

if __name__ == "__main__":
    main()
