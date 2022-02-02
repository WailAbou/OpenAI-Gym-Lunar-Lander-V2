from dataclasses import dataclass
from typing import Any


@dataclass
class Transition:
    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool
