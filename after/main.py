import pathlib

import torch

from ds.dataset import create_dataloader
from ds.models import LinearNet
from ds.epoch import run_epoch
from ds.tensorboard import TensorboardExperiment

# Hyperparameters
EPOCH_COUNT = 20
LR = 5e-5
BATCH_SIZE = 128
LOG_PATH = "./runs"

# Data configuration
DATA_DIR = "./data/raw"
TEST_DATA = pathlib.Path(f"{DATA_DIR}/t10k-images-idx3-ubyte.gz")
TEST_LABELS = pathlib.Path(f"{DATA_DIR}/t10k-labels-idx1-ubyte.gz")
TRAIN_DATA = pathlib.Path(f"{DATA_DIR}/train-images-idx3-ubyte.gz")
TRAIN_LABELS = pathlib.Path(f"{DATA_DIR}/train-labels-idx1-ubyte.gz")


def main():

    # Model and Optimizer
    model = LinearNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Create the data loaders
    test_loader = create_dataloader(BATCH_SIZE, TEST_DATA, TEST_LABELS)
    train_loader = create_dataloader(BATCH_SIZE, TRAIN_DATA, TRAIN_LABELS)

    # Setup the experiment tracker
    experiment_tracker = TensorboardExperiment(log_path=LOG_PATH)

    # Run the epochs
    for epoch_id in range(EPOCH_COUNT):
        train_avg_accuracy, test_avg_accuracy = run_epoch(
            epoch_id,
            test_loader,
            train_loader,
            model,
            optimizer,
            experiment_tracker)

        # Compute Average Epoch Metrics
        summary = ", ".join(
            [
                f"[Epoch: {epoch_id + 1}/{EPOCH_COUNT}]",
                f"Test Accuracy: {test_avg_accuracy: 0.4f}",
                f"Train Accuracy: {train_avg_accuracy: 0.4f}",
            ]
        )
        print("\n" + summary + "\n")

        # Flush the experiment tracker after every epoch for live updates
        experiment_tracker.flush()


if __name__ == "__main__":
    main()
