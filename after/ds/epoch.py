from typing import Any, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from ds.metrics import Metric
from ds.tracking import ExperimentTracker, Stage

def EpochException(Exception):
    pass


def _run_batch(
        x: Any,
        y: Any,
        y_true_batches: list[list[Any]],
        y_pred_batches: list[list[Any]],
        model: torch.nn.Module,
        accuracy_metric,
        loss_fn:torch.nn.modules.module.Module) -> Tuple[float, float]:

    batch_size: int = x.shape[0]
    prediction = model(x)
    loss = loss_fn(prediction, y)

    # Compute Batch Validation Metrics
    y_np = y.detach().numpy()
    y_prediction_np = np.argmax(prediction.detach().numpy(), axis=1)
    batch_accuracy: float = accuracy_score(y_np, y_prediction_np)
    accuracy_metric.update(batch_accuracy, batch_size)

    y_true_batches += [y_np]
    y_pred_batches += [y_prediction_np]
    return loss, batch_accuracy


def _run_epoch_stage(
        description: str,
        stage: Stage,
        epoch_id: int,
        loader: DataLoader[Any],
        model: torch.nn.Module,
        experiment_tracker: ExperimentTracker,
        optimizer: Optional[torch.optim.Optimizer]) -> float:
    
    experiment_tracker.set_stage(stage)

    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")

    accuracy_metric = Metric()

    model.train(stage is Stage.TRAIN)

    y_true_batches: list[list[Any]] = []
    y_pred_batches: list[list[Any]] = []

    for batch_count, (x, y) in enumerate(tqdm(loader, description, ncols=120)):
        loss, batch_accuracy = _run_batch(
            x, y, 
            y_true_batches,
            y_pred_batches,
            model,
            accuracy_metric,
            loss_fn)

        experiment_tracker.add_batch_metric("accuracy", batch_accuracy, batch_count)

        if stage == Stage.VALIDATE:
            if optimizer is None:
                raise EpochException("Optimizer not provided for Validation Stage")
            # Reverse-mode AutoDiff (backpropagation)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Log Epoch Metrics for this stage
    if stage == Stage.TRAIN:
        experiment_tracker.add_epoch_metric("accuracy", accuracy_metric.average, epoch_id)
    elif stage == Stage.VALIDATE:
        # Log Validation Epoch Metrics
        experiment_tracker.add_epoch_metric("accuracy", accuracy_metric.average, epoch_id)
        experiment_tracker.add_epoch_confusion_matrix(
            y_true_batches, y_pred_batches, epoch_id)
    else:
        raise EpochException("Unsupported stage")

    return accuracy_metric.average

def run_epoch(
        epoch_id: int,
        test_loader: DataLoader[Any],
        train_loader: DataLoader[Any],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        experiment_tracker: ExperimentTracker
        ) -> Tuple[float, float]:

    # Training Loop
    test_avg_accuracy = _run_epoch_stage(
        "Train Batches",
        stage=Stage.TRAIN,
        epoch_id=epoch_id,
        loader=train_loader,
        model=model,
        experiment_tracker=experiment_tracker,
        optimizer=None)

    # Testing/Validation Loop
    train_avg_accuracy = _run_epoch_stage(
        "Validation Batches",
        stage=Stage.VALIDATE,
        epoch_id=epoch_id,
        loader=test_loader,
        model=model,
        experiment_tracker=experiment_tracker,
        optimizer=optimizer)

    return train_avg_accuracy, test_avg_accuracy
