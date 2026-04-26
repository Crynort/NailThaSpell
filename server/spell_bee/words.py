import random
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Word:
    text: str


EASY: List[Word] = [
    Word("apple"),
    Word("river"),
    Word("happy"),
    Word("table"),
    Word("cloud"),
]

MEDIUM: List[Word] = [
    Word("rhythm"),
    Word("island"),
    Word("knight"),
    Word("breeze"),
    Word("plumber"),
]

HARD: List[Word] = [
    Word("bouquet"),
    Word("silhouette"),
    Word("conscience"),
    Word("rendezvous"),
    Word("entrepreneur"),
]


def get_word(round_index: int) -> Word:
    """Pick a word based on which round we are in."""
    if round_index < 2:
        bucket = EASY
    elif round_index < 4:
        bucket = MEDIUM
    else:
        bucket = HARD
    return random.choice(bucket)
