"""Serve the H1-2's hardware estop box over TCP, for the estop package's estop_node.

The real box is an Arduino on USB serial (core_ws/src/estop/arduino/estop_serial.ino)
that prints "ESTOP,BOOT" once, then "ESTOP,0" (clear) or "ESTOP,1" (triggered) on every
loop. This server speaks the same lines to every client on 127.0.0.1:<port>, which
estop_node opens as serial_port socket://127.0.0.1:<port> (pyserial's serial_for_url),
so the sim bringup runs the real estop_node and golem_safety's estop path unchanged.

The port is estop_port() of the sim's ROS_DOMAIN_ID, so a sim on another domain never
answers this domain's estop_node. The ENTER key, on the sim's terminal or in the
viewer, toggles the button between clear and triggered.

SAFETY: golem_safety latches its estop until restart on a triggered sample, on an
unplugged box and on a box silent for its estop.timeout_seconds; the server streams on
the wall clock from its own thread, so a stalled physics loop still reads as a live box,
and a stopped sim reads as an unplugged one.
"""
import os
import select
import socket
import sys
import threading

# First port of the estop servers; the sim on ROS domain d listens on BASE_PORT + d.
BASE_PORT = 15700
# Hz of lines sent to each client: estop_node treats a box silent for its
# line_stale_timeout_sec (0.1 s) as stale.
LINE_HZ = 50.0
# GLFW's key code for ENTER, as the MuJoCo viewer passes it to a key callback.
GLFW_KEY_ENTER = 257


def estop_port(domain_id: str | None = None) -> int:
    """Return the estop TCP port of a ROS domain (ROS_DOMAIN_ID when domain_id is None)."""
    raw = os.environ.get("ROS_DOMAIN_ID", "") if domain_id is None else domain_id
    return BASE_PORT + int(raw.strip() or "1")


class EstopServer:
    """Stream the simulated estop box's lines to every connected estop_node."""

    def __init__(self, port: int, tag: str):
        """Bind the port; raise OSError when another process holds it.

        tag prefixes the log lines (e.g. "golem_sim").
        """
        self._tag = tag
        self._lock = threading.Lock()
        self._triggered = False
        self._clients: list[socket.socket] = []
        self._stop = threading.Event()
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(("127.0.0.1", port))
        self._listener.listen(4)
        self._listener.settimeout(0.1)
        self.port = port
        self._threads = [
            threading.Thread(target=self._accept_loop, daemon=True, name="estop_accept"),
            threading.Thread(target=self._send_loop, daemon=True, name="estop_send"),
        ]
        if sys.stdin is not None and sys.stdin.isatty():
            self._threads.append(
                threading.Thread(target=self._stdin_loop, daemon=True, name="estop_stdin"))

    def start(self) -> None:
        """Start serving, and reading ENTER from the terminal when it is one."""
        for thread in self._threads:
            thread.start()
        keys = "ENTER (terminal or viewer)" if len(self._threads) == 3 else "ENTER (viewer)"
        print(f"[{self._tag}] estop box on socket://127.0.0.1:{self.port}: clear; "
              f"{keys} toggles it", flush=True)

    def toggle(self) -> None:
        """Press the button if it is clear, release it if it is pressed."""
        with self._lock:
            self._triggered = not self._triggered
            state = self._triggered
        print(f"[{self._tag}] estop {'TRIGGERED' if state else 'released'}", flush=True)

    def key_callback(self, keycode: int) -> None:
        """Toggle the button on ENTER (a MuJoCo viewer key callback)."""
        if keycode == GLFW_KEY_ENTER:
            self.toggle()

    def shutdown(self) -> None:
        """Stop serving and close every connection, which estop_node reads as unplugged."""
        self._stop.set()
        for thread in self._threads:
            if thread.name != "estop_stdin":
                thread.join(timeout=1.0)
        with self._lock:
            clients, self._clients = self._clients, []
        for client in clients:
            client.close()
        self._listener.close()

    def _accept_loop(self) -> None:
        """Accept estop_node's connections; each gets ESTOP,BOOT first, as at power-up."""
        while not self._stop.is_set():
            try:
                client, _ = self._listener.accept()
            except (socket.timeout, OSError):
                continue
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                client.sendall(b"ESTOP,BOOT\r\n")
            except OSError:
                client.close()
                continue
            with self._lock:
                self._clients.append(client)

    def _send_loop(self) -> None:
        """Send the button's state to every client at LINE_HZ, on the wall clock."""
        period = 1.0 / LINE_HZ
        while not self._stop.wait(period):
            with self._lock:
                line = b"ESTOP,1\r\n" if self._triggered else b"ESTOP,0\r\n"
                clients = list(self._clients)
            dead = []
            for client in clients:
                try:
                    client.sendall(line)
                except OSError:
                    dead.append(client)
            if dead:
                with self._lock:
                    self._clients = [c for c in self._clients if c not in dead]
                for client in dead:
                    client.close()

    def _stdin_loop(self) -> None:
        """Toggle the button on each line read from the terminal."""
        while not self._stop.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], 0.2)
            if not ready:
                continue
            if not sys.stdin.readline():
                return
            self.toggle()


def start_estop_server(tag: str) -> EstopServer:
    """Start the estop server on this domain's port; exit the sim when the port is taken."""
    port = estop_port()
    try:
        server = EstopServer(port, tag)
    except OSError as exc:
        raise SystemExit(
            f"[{tag}] cannot serve the estop on 127.0.0.1:{port} ({exc}): another sim "
            f"on ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '')!r} holds it") from exc
    server.start()
    return server


__all__ = ["BASE_PORT", "EstopServer", "estop_port", "start_estop_server"]
