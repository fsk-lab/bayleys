import time
from logging import Logger
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl


def format_time(seconds: float) -> str:
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class LoggerCallback(TrainerCallback):
    """
    A custom Huggingface TrainerCallback that logs training progress to a provided logger.
    """

    def __init__(self, logger: Logger):
        self.logger = logger
        self._start_time = None
        self._last_time = None
        self._last_step = None

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        now = time.perf_counter()
        self._start_time = now
        self._last_time = now
        self._last_step = state.global_step

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if not logs:
            return

        if not state.is_world_process_zero:
            return

        now = time.perf_counter()
        step = state.global_step
        total_time = now - self._start_time
        speed = (step - self._last_step) / (now - self._last_time)

        log_message = f"Step {step} (Epoch {state.epoch:.2f}) - "
        log_message += f"Total Time: {format_time(total_time)} - Speed: {speed:.2f} steps/s - "
        for key, value in logs.items():
            if key in ("epoch", ):
                continue
            if isinstance(value, float):
                log_message += f"{key}: {value:.4e} - "
            else:
                log_message += f"{key}: {value} - "

        self.logger.info(log_message)