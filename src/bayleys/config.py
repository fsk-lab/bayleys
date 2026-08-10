import os
from typing import Optional, Literal
from dataclasses import dataclass, replace, fields
from pathlib import Path
from threading import RLock
import logging
import torch

from .utils import AbstractConfig


@dataclass(frozen=True)
class BayleysGlobalConfig:
    """
    Global configuration for the Bayleys library, which can be accessed throughout the package. Cannot be directly
    instantiated or modified by users.
    """
    device: Literal["cpu", "gpu"] = "cpu"
    cache_dir: Optional[str | Path] = None
    tmp_dir: Optional[str | Path] = None

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_file: Optional[str | Path] = None

    def __post_init__(self):
        """
        Post-initialization to set up device, cache directory, and logging.
        """
        if self.device != "cpu":
            # Auto-detect if a GPU is available, and set the torch device accordingly
            if torch.cuda.is_available():
                object.__setattr__(self, 'device', 'cuda')
            elif torch.mps.is_available():
                object.__setattr__(self, 'device', 'mps')

        # Set the cache directory
        if self.cache_dir is None:
            cwd = Path.cwd()
            object.__setattr__(self, 'cache_dir', str(cwd / "data" / "cache"))
        else:
            cache_path = Path(self.cache_dir)
            if not cache_path.is_absolute():
                cache_path = Path.cwd() / cache_path
            object.__setattr__(self, 'cache_dir', str(cache_path))

        # Set the temporary directory
        if self.tmp_dir is None:
            cwd = Path.cwd()
            object.__setattr__(self, 'tmp_dir', str(cwd / "data" / "tmp"))
        else:
            tmp_path = Path(self.tmp_dir)
            if not tmp_path.is_absolute():
                tmp_path = Path.cwd() / tmp_path
            object.__setattr__(self, 'tmp_dir', str(tmp_path))

        # Define the logger
        logger = logging.getLogger("bayleys")
        logging.shutdown(), logger.handlers.clear(), logging.root.handlers.clear()
        log_level = getattr(logging, self.log_level.upper(), logging.INFO)
        logger.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        if self.log_file is not None:
            if Path(self.log_file).is_absolute():
                file_handler = logging.FileHandler(self.log_file)
            else:
                file_handler = logging.FileHandler(Path.cwd() / self.log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # Set the number of CPU cores, if applicable
        if self.device == "cpu":
            try:
                num_cpus = len(os.sched_getaffinity(0))
            except AttributeError:
                num_cpus = os.cpu_count() or 1
            try:
                torch.set_num_threads(num_cpus)
                torch.set_num_interop_threads(1)
                logger.info(f"Set the number of CPU threads for PyTorch to {num_cpus}.")
            except Exception as e:
                logger.warning(f"Failed to set the number of CPU threads for PyTorch: {e}")


# Singleton instance and lock for thread-safe access
LOCK = RLock()
CONFIG: Optional[BayleysGlobalConfig] = None


def get_config() -> BayleysGlobalConfig:
    """
    Retrieves the global Bayleys configuration. If it has not been initialized yet, it creates a default configuration.

    Returns:
        BayleysGlobalConfig: The global configuration object.
    """
    global CONFIG
    with LOCK:
        if CONFIG is None:
            CONFIG = BayleysGlobalConfig()
        return CONFIG


def set_config(new_config: BayleysGlobalConfig):
    """
    Sets the global Bayleys configuration to a new configuration.

    Args:
        new_config (BayleysGlobalConfig): The new configuration to set.
    """
    global CONFIG
    with LOCK:
        CONFIG = new_config


@dataclass
class BayleysConfig(AbstractConfig):
    """
    Base configuration class for Bayleys components. Used to store and pass around common configuration options, and
    modify the `BayleysGlobalConfig` settings.
    """
    device: Optional[Literal["cpu", "gpu"]] = None
    cache_dir: Optional[str | Path] = None
    tmp_dir: Optional[str | Path] = None
    log_level: Optional[Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]] = None
    log_file: Optional[str | Path] = None

    def apply(self):
        """
        Applies the configuration settings to the global Bayleys configuration.
        """
        replacements = {field.name: getattr(self, field.name) for field in fields(BayleysConfig)}
        replacements = {k: v for k, v in replacements.items() if v is not None}

        with LOCK:
            cfg = get_config()
            new_cfg = replace(cfg, **replacements)
            set_config(new_cfg)
