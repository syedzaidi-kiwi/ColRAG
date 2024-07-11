from abc import ABC, abstractmethod
from typing import Any

class BasePlugin(ABC):
    @abstractmethod
    def can_handle(self, file_extension: str) -> bool:
        pass

    @abstractmethod
    def read_file(self, file_path: str) -> Any:
        pass