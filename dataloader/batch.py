"""
General Batch and BatchFetcher classes for parallel data loading.
Used by DataLoader, created by DataLoaderFactory.
"""

from abc import ABC, abstractmethod
from studies import Study, Recording
import torch
import mne


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

    def _segment_brain(
        self,
        recording: Recording,
        raw: mne.io.Raw,
        brain_window_timestamps: list[tuple[float, float]],
    ) -> dict[str, torch.Tensor]:
        """
        Slice the brain data into segments based on the window time stamps.
        Returns dictionary of brain segments for each frequency band, containing
        tensor of shape [B, C, T] where B (windows), C (channels), T (time steps).
        """

        results = recording.pre_process_brain(
            raw=raw,
            notch_filter=self.notch_filter,
            frequency_bands=self.frequency_bands,
            scaling=self.scaling,
            brain_clipping=self.brain_clipping,
            baseline_window=self.baseline_window,
            new_freq=self.new_freq,
            n_jobs=self.n_jobs,
        )
        # np array of time stamps corresponsing to brain data
        times = torch.from_numpy(results[list(self.frequency_bands.keys())[0]].times)

        n_windows = len(brain_window_timestamps)
        data, brain_segments = {}, {}

        # Initialize output arrays for each band, of shape [B, C, T] (with padding)
        for band in self.frequency_bands.keys():

            data[band] = torch.from_numpy(results[band].get_data())  # [C, T]
            n_channels = data[band].shape[0]
            brain_segments[band] = torch.zeros(
                (n_windows, n_channels, self.window_size * self.new_freq),
                dtype=torch.float32,
            )

        # Extract windows for all bands
        for i, (start_time, end_time) in enumerate(brain_window_timestamps):

            mask = (times >= start_time) & (times <= end_time)

            for band in self.frequency_bands.keys():
                window = data[band][:, mask]  # Extract window
                brain_segments[band][i] = window[
                    :, : self.window_size * self.new_freq
                ]  # Truncate to [C, T]

        return brain_segments
