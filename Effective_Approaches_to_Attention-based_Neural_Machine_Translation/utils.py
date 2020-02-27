import json
import torch
from pathlib import Path
from typing import Union


class Config:
    """Config class"""

    def __init__(self, json_path_or_dict: Union[str, dict]) -> None:
        """Instantiating Config class
        Args:
            json_path_or_dict (Union[str, dict]): filepath of config or dictionary which has attributes
        """
        if isinstance(json_path_or_dict, dict):
            self.__dict__.update(json_path_or_dict)
        else:
            with open(json_path_or_dict, mode="r") as io:
                params = json.loads(io.read())
            self.__dict__.update(params)

    def save(self, json_path: Union[str, Path]) -> None:
        """Saving config to json_path
        Args:
            json_path (Union[str, Path]): filepath of config
        """
        with open(json_path, mode="w") as io:
            json.dump(self.__dict__, io, indent=4)

    def update(self, json_path_or_dict) -> None:
        """Updating Config instance
        Args:
            json_path_or_dict (Union[str, dict]): filepath of config or dictionary which has attributes
        """
        if isinstance(json_path_or_dict, dict):
            self.__dict__.update(json_path_or_dict)
        else:
            with open(json_path_or_dict, mode="r") as io:
                params = json.loads(io.read())
            self.__dict__.update(params)

    @property
    def dict(self) -> dict:
        return self.__dict__


class CheckpointManager:
    """CheckpointManager class"""

    def __init__(self, model_dir: Union[str, Path]) -> None:
        """Instantiating CheckpointManager class
        Args:
            model_dir (Union[str, Path]): directory path for saving a checkpoint
        """
        if not isinstance(model_dir, Path):
            model_dir = Path(model_dir)

        if not model_dir.exists():
            model_dir.mkdir(parents=True)

        self._model_dir = model_dir

    def save_checkpoint(self, state: dict, filename: str) -> None:
        """Saving a checkpoint
        Args:
            state (dict): a checkpoint
            filename (str): the filename of a checkpoint
        """
        torch.save(state, self._model_dir / filename)

    def load_checkpoint(self, filename: str, device: torch.device = None) -> dict:
        """Loading a checkpoint
        Args:
            filename (str): the filename of a checkpoint
            device (torch.device): device where a checkpoint will be stored
        Returns:
            state (dict): a checkpoint
        """
        device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        state = torch.load(self._model_dir / filename, map_location=device)
        return state


class SummaryManager:
    """SummaryManager class"""

    def __init__(self, model_dir: Union[str, Path]) -> None:
        if not isinstance(model_dir, Path):
            model_dir = Path(model_dir)
        if not model_dir.exists():
            model_dir.mkdir(parents=True)

        self._model_dir = model_dir
        self._summary = {}

    def save(self, filename: str) -> None:
        """Saving a summary to model_dir
        Args:
            filename (str): the filename of a summary
        """
        with open(self._model_dir / filename, mode="w") as io:
            json.dump(self._summary, io, indent=4)

    def load(self, filename) -> None:
        """Loading a summary from model_dir
        Args:
            filename (str): the filename of a summary
        """
        with open(self._model_dir / filename, mode="r") as io:
            metric = json.loads(io.read())
        self.update(metric)

    def update(self, summary: dict) -> None:
        """Updating a summary
        Args:
            summary (dict): a summary
        """
        self._summary.update(summary)

    def reset(self) -> None:
        """Resetting a summary"""
        self._summary = {}

    @property
    def summary(self):
        return self._summary
