"""Spell Bee bot package."""

from spell_bee.processor import SpellBeeProcessor
from spell_bee.spelling_parser import is_correct, parse_spelling
from spell_bee.words import EASY, HARD, MEDIUM, Word, get_word

__all__ = [
    "SpellBeeProcessor",
    "Word",
    "EASY",
    "MEDIUM",
    "HARD",
    "get_word",
    "parse_spelling",
    "is_correct",
]
