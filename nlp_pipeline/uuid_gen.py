"""
A UUIDGenerator which can be used independently or as a global object to produce
pseudorandom or reproducible UUIDs.

This is used for UUID generation in stages for adding elements to record mappings,
and on DataFrame deconstruction before all stages.
"""

import random
from typing import Generator
import uuid


class UUIDGenerator:
    """Reproducable UUID generator with seeded randomness."""

    _random_state: int | None
    _rng: random.Random

    def __init__(self, random_state: int | None = None):
        self._random_state = random_state
        self._rng = random.Random(random_state)

    def reset(self, random_state: int | None = None):
        """
        Resets the underlying Random object to its provided random_state. If
        no random_state is provided, the current random_state is used.
        """
        if random_state is not None:
            self._random_state = random_state
        self._rng.seed(self._random_state)

    def next(self) -> str:
        """Gets the next UUID string from the stored Random object."""
        return uuid.UUID(int=self._rng.getrandbits(128), version=4).hex


# Global default UUIDGenerator instance with constant seed
_default_uuid_seed = 42
_default_uuid_generator = UUIDGenerator(_default_uuid_seed)


def default_uuid_generator():
    """Gets the global default UUIDGenerator instance"""
    return _default_uuid_generator
