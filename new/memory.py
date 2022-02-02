from collections import deque
from typing import Any, List

from attr import dataclass


from dataclasses import dataclass
from transition import Transition
import random


@dataclass
class Memory:
    size: int
    content: deque = deque([])

    def sample(self, batch_size: int) -> List[Transition]:
        batch_size = min(batch_size, len(self.content))
        return random.sample(self.content, batch_size)

    def record(self, transition: Transition) -> None:
        self.content.append(transition)
        if len(self.content) > self.size:
            self.content.popleft()
