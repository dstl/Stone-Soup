# -*- coding: utf-8 -*-
"""Base classes for use with File based readers."""
from pathlib import Path

from .base import Reader
from ..base import Property


class FileReader(Reader):
    """Base class for file based readers.

    Parameters
    ----------
    path : :class:`pathlib.Path` or str
        Path to file to be opened. Str will be converted to path.
    """
    path = Property(Path)

    def __init__(self, path, *args, **kwargs):
        if not isinstance(path, Path):
            path = Path(path)  # Ensure Path
        super().__init__(path, *args, **kwargs)
        self._file = None

    def __del__(self):
        self._file.close()


class BinaryFileReader(FileReader):
    """Base class for binary file readers.

    Parameters
    ----------
    path : :class:`pathlib.Path` or str
        Path to file to be opened. Str will be converted to path.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._file = self.path.open('rb')


class TextFileReader(FileReader):
    """Base class for text file readers.

    Parameters
    ----------
    path : :class:`pathlib.Path` or str
        Path to file to be opened. Str will be converted to path.
    encoding : str, optional
        File encoding. Must be valid coding. Default 'utf-8'
    """
    encoding = Property(str, default="utf-8")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._file = self.path.open('r', encoding=self.encoding)
