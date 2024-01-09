# Custom Callback 클래스 정의
from transformers import TrainerCallback


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, early_stopping_patience, early_stopping_threshold, early_stopping_metric, early_stopping_metric_minimize):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_metric_minimize = early_stopping_metric_minimize
        self.best_metric = float("inf") if self.early_stopping_metric_minimize else float("-inf")
        self.waiting_steps = 0

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        current_metric = logs.get(self.early_stopping_metric, None)
        if current_metric is not None:
            if (self.early_stopping_metric_minimize and current_metric < self.best_metric) or (not self.early_stopping_metric_minimize and current_metric > self.best_metric):
                self.best_metric = current_metric
                self.waiting_steps = 0
            else:
                self.waiting_steps += 1

                if self.waiting_steps >= self.early_stopping_patience:
                    print(f"Early stopping triggered after {self.waiting_steps} steps without improvement.")
                    control.should_training_stop = True
