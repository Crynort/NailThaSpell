"""Hardcoded word list for the Spell Bee game.

Words are organized into three difficulty buckets. The game starts with easy
words and progresses to harder ones as the user gets answers correct.

Each entry includes a phonetic pronunciation hint that the bot can use to
disambiguate homophones (e.g. their/there/they're) when speaking the word.
"""

import random
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Word:
    """A single spell-bee word."""

    text: str           # The actual spelling
    definition: str     # Short definition the bot can read out as a hint
    example: str        # Example sentence using the word

    @property
    def length(self) -> int:
        return len(self.text)


# --- Difficulty tiers -------------------------------------------------------
# Kept short and recognizable so the demo is reliable over a noisy mic.
EASY: List[Word] = [
    Word("apple", "a common round fruit, usually red or green",
         "She packed an apple in her lunch."),
    Word("river", "a large natural stream of water",
         "We swam in the river all afternoon."),
    Word("happy", "feeling or showing pleasure",
         "The children were happy at the party."),
    Word("table", "a piece of furniture with a flat top",
         "Please set the table for dinner."),
    Word("cloud", "a visible mass of water droplets in the sky",
         "There is not a cloud in the sky today."),
]

MEDIUM: List[Word] = [
    Word("rhythm", "a strong, regular, repeated pattern of sound",
         "The drummer kept a steady rhythm."),
    Word("island", "a piece of land surrounded by water",
         "They sailed to a small island."),
    Word("knight", "a man given a special honor by a king or queen",
         "The knight rode a white horse."),
    Word("breeze", "a gentle wind",
         "A cool breeze came through the window."),
    Word("plumber", "a person who fits and repairs water pipes",
         "We called the plumber to fix the sink."),
]

HARD: List[Word] = [
    Word("bouquet", "an attractively arranged bunch of flowers",
         "She carried a bouquet of roses."),
    Word("silhouette", "the dark outline of someone or something against a brighter background",
         "I saw his silhouette in the doorway."),
    Word("conscience", "a person's moral sense of right and wrong",
         "His conscience would not let him lie."),
    Word("rendezvous", "a meeting at an agreed time and place",
         "They had a rendezvous at the cafe."),
    Word("entrepreneur", "a person who sets up and runs a business",
         "She is a successful entrepreneur."),
]


def get_word(round_index: int) -> Word:
    """Pick a word based on which round we are in.

    Rounds 1-3 → easy, 4-6 → medium, 7+ → hard. Within each tier we select
    a random word from the bucket.
    """
    if round_index < 3:
        bucket = EASY
    elif round_index < 6:
        bucket = MEDIUM
    else:
        bucket = HARD
    return random.choice(bucket)
