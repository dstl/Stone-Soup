import functools
import logging

LOG_ID = 'stonesoup'
LOG = logging.getLogger(LOG_ID)


class LogFunction:
    """Logging decorator that allows you to log with a
    specific logger.
    """
    # Customize these messages
    ENTRY_MESSAGE = 'Entering {}'
    EXIT_MESSAGE = 'Exiting {}'

    def __init__(self, logger=None):
        self.logger = logger

    def __call__(self, func):
        """Returns a wrapper that wraps func.
        The wrapper will log the entry and exit points of the function
        with logging.INFO level.
        """
        # set logger if it was not set earlier
        if not self.logger:
            logging.basicConfig()
            self.logger = LOG

        @functools.wraps(func)
        def wrapper(*args, **kwds):
            self.logger.debug(self.ENTRY_MESSAGE.format(func.__name__))
            f_result = func(*args, **kwds)
            self.logger.debug(self.EXIT_MESSAGE.format(func.__name__))
            return f_result
        return wrapper
