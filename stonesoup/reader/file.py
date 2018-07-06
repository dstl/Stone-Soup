# -*- coding: utf-8 -*-
"""Base classes for use with File based readers."""
from pathlib import Path

from .base import Reader
from ..base import Property


class FileReader(Reader):
    """Base class for file based readers."""
    path = Property(
        Path,
        doc="Path to file to be opened. Str will be converted to path.")

    def __init__(self, path, *args, **kwargs):
        if not isinstance(path, Path):
            path = Path(path)  # Ensure Path
        super().__init__(path, *args, **kwargs)
        self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if getattr(self, '_file', None):
            self._file.close()

    def __del__(self):
        self.__exit__()


class BinaryFileReader(FileReader):
    """Base class for binary file readers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._file = self.path.open('rb')


class TextFileReader(FileReader):
    """Base class for text file readers."""
    encoding = Property(
        str, default="utf-8",
        doc="File encoding. Must be valid coding. Default 'utf-8'.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._file = self.path.open('r', encoding=self.encoding)
