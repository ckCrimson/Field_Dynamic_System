import sys
from typing import Any, Dict
from .base import IPolicy


class KeyboardPolicy(IPolicy):
    """
    A policy driven by live human keyboard input.
    """

    def __init__(self):
        self._action_map = {
            'left': -1, 'a': -1,
            'right': 1, 'd': 1,
            'down': 0, 's': 0
        }

    def _get_keypress(self) -> str:
        """Cross-platform hidden keystroke catcher."""
        try:
            import msvcrt
            key = msvcrt.getch()
            if key in (b'\xe0', b'\x00'):
                return {'H': 'up', 'P': 'down', 'K': 'left', 'M': 'right'}.get(msvcrt.getch().decode('utf-8'), '')
            return key.decode('utf-8').lower()
        except ImportError:
            import tty, termios
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
                if ch == '\x1b':
                    sys.stdin.read(1)  # skip '['
                    return {'A': 'up', 'B': 'down', 'C': 'right', 'D': 'left'}.get(sys.stdin.read(1), '')
                return ch.lower()
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def get_action(self, state: Any) -> Dict[str, Any]:
        """Blocks and waits for the user to press a valid key."""
        while True:
            key = self._get_keypress()

            if key == 'q':
                return {"quit": True, "action_name": "Quit"}

            if key in self._action_map:
                action_id = self._action_map[key]
                name = "Left" if action_id == -1 else "Right" if action_id == 1 else "Rest"
                return {"action_id": action_id, "quit": False, "action_name": name}