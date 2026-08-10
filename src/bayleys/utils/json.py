from abc import ABC, abstractmethod
from pathlib import Path
import json


class JSONSerializable(ABC):
    """
    Mixin class to provide JSON serialization and deserialization methods.
    """

    @abstractmethod
    def to_dict(self, **kwargs) -> dict:
        """
        Convert the object to a dictionary representation.

        Returns:
            dict: Dictionary representation of the object.
        """
        raise NotImplementedError()

    def to_json(self, file: str | Path):
        """
        Serialize the object to a JSON file.

        Args:
            file (str | Path): Path to the JSON file where the object will be saved.
        """
        with open(file, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict, **kwargs):
        """
        Create an object from a dictionary representation.

        Args:
            data (dict): Dictionary representation of the object.

        Returns:
            JSONSerializable: An instance of the class.
        """
        raise NotImplementedError()

    @classmethod
    def from_json(cls, file: str | Path, **kwargs):
        """
        Deserialize an object from a JSON file.

        Args:
            file (str | Path): Path to the JSON file from which the object will be loaded.
            **kwargs: Additional keyword arguments to pass to the from_dict method.

        Returns:
            JSONSerializable: An instance of the class.
        """
        with open(file, "r") as f:
            data = json.load(f)
        return cls.from_dict(data, **kwargs)
