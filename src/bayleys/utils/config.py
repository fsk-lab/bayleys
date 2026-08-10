from typing import Union, Literal, get_origin
from types import UnionType
from pathlib import Path
from abc import ABC
from dataclasses import dataclass
import json
from hashlib import sha256

from .json import JSONSerializable


@dataclass
class AbstractConfig(JSONSerializable, ABC):
    """
    Abstract base class for configuration classes.
    """

    def to_dict(self) -> dict:
        """
        Converts the configuration object to a dictionary. Recursively converts nested configuration objects to
        dictionaries, too.

        Returns:
            dict: Dictionary representation of the configuration object.
        """
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, AbstractConfig):
                config_dict[key] = value.to_dict()
            elif isinstance(value, Path):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        return config_dict

    @classmethod
    def from_dict(cls, data: dict, **kwargs):
        """
        Creates a configuration object from a dictionary. Recursively creates nested configuration objects from
        dictionaries, too.

        Args:
            data (dict): Dictionary representation of the configuration object.

        Returns:
            AbstractConfig: Configuration object created from the dictionary.
        """
        for name, field in cls.__dataclass_fields__.items():
            if get_origin(field.type) in (Union, Literal, UnionType):
                continue
            if issubclass(field.type, AbstractConfig):
                if data.get(name) is not None:
                    data[name] = field.type.from_dict(data[name])
                else:
                    data[name] = None

        return cls(**data)

    @property
    def unique_id(self) -> str:
        """
        Generates a unique identifier for the configuration object based on its class name and attributes.

        Returns:
            str: Unique identifier for the configuration object.
        """
        payload = json.dumps(self.to_dict(), sort_keys=True).encode('utf-8')
        hash_digest = sha256(payload).hexdigest()[:8]
        return hash_digest
