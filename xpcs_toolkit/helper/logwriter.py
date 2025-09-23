from collections.abc import Callable


# https://stackoverflow.com/questions/19425736
# /how-to-redirect-stdout-and-stderr-to-logger-in-python
class LoggerWriter:
    def __init__(self, level: Callable[[str], None]) -> None:
        # self.level is really like using log.debug(message)
        # at least in my case
        self.level = level

    def write(self, message: str) -> None:
        # if statement reduces the amount of newlines that are
        # printed to the logger
        if message != "\n":
            self.level(message)

    def flush(self) -> None:
        # create a flush method so things can be flushed when
        # the system wants to. Nothing needs to be logged during flush.
        pass
