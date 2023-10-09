"""Base classes for use with File based readers."""
from urllib.parse import ParseResult, urlparse

from .base import Reader
from ..base import Property


class UrlReader(Reader):
    """Base class for url based readers to read files from abstract URLs"""

    url = Property(
        ParseResult,
        doc="URL path to be parsed. Str will be converted to url.")

    def __init__(self, url, *args, **kwargs):
        if not isinstance(url, ParseResult):
            url = urlparse(url)  # Ensure Path
        super().__init__(url, *args, **kwargs)
