"""
General Batch and BatchFetcher classes for parallel data loading.
Used by DataLoader, created by DataLoaderFactory.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import ray

from studies import Study, Recording


# @dataclass
class Batch(ABC):
    pass


# @ray.remote
class BatchFetcher(ABC):
    def __init__(
        self,
        notch_filter: bool,
        frequency_bands: dict[str, tuple[float, float]],
        scaling: str,
        brain_clipping: int,
        baseline_window: float,
        new_freq: int,
    ):
        super(BatchFetcher, self).__init__()
        # Pre-processing attributes shared by all kinds of studies and tasks
        self.notch_filter = notch_filter
        self.frequency_bands = frequency_bands
        self.scaling = scaling
        self.brain_clipping = brain_clipping
        self.baseline_window = baseline_window
        self.new_freq = new_freq

    @abstractmethod
    def fetch(self, recording: Recording, cache: bool) -> Batch:
        """To be implemented for each Batch type. Layout is handled by model."""
        pass
