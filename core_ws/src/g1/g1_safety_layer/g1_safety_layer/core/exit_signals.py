'''SIGINT, SIGTERM and SIGHUP as a request to hand control back before exiting'''

import signal
import threading

# Ctrl-C, ros2 launch shutting down (SIGINT, then SIGTERM after 5 s), docker
# stop (SIGTERM to PID 1) and a closed terminal or SSH session (SIGHUP, whose
# default would end the process at once); SIGKILL cannot be caught
EXIT_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


class ExitRequest:
    '''Record the first exit signal; later ones change nothing'''
    def __init__(self) -> None:
        self.signal_name: str | None = None
        self._event = threading.Event()

    def install(self) -> None:
        '''Install the handlers (main thread only)'''
        for signum in EXIT_SIGNALS:
            signal.signal(signum, self._handle)

    def is_set(self) -> bool:
        '''Whether an exit signal has arrived'''
        return self._event.is_set()

    def _handle(self, signum: int, _frame) -> None:
        if self.signal_name is None:
            self.signal_name = signal.Signals(signum).name
        self._event.set()
