"""
General Batch and BatchFetcher classes for parallel data loading.
Used by DataLoader, created by DataLoaderFactory.
"""

from abc import ABC, abstractmethod
import time
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
        delay: float,
        n_jobs: int = 1,
    ):
        super(BatchFetcher, self).__init__()
        # Pre-processing attributes shared by all kinds of studies and tasks
        self.notch_filter = notch_filter
        self.frequency_bands = frequency_bands
        self.scaling = scaling
        self.brain_clipping = brain_clipping
        self.baseline_window = baseline_window
        self.new_freq = new_freq
        self.delay = delay
        self.n_jobs = n_jobs

    @abstractmethod
    def fetch(self, recording: Recording, cache: bool) -> Batch:
        """To be implemented for each Batch type. Layout is handled by model."""
        pass

    # @abstractmethod
    # def load_from_cache(self, recording: Recording) -> Batch:
    #     """Load a batch from cache, if available."""
    #     pass

    def segment_brain_mne(
        self,
        recording: Recording,
        raw: mne.io.Raw,
        brain_window_timestamps: list[tuple[float, float]],
        batch_size: int,
        cache: bool,
    ) -> dict[str, torch.Tensor]:
        """Different to segment_brain_tensor, this function alsopre-processes MNE.
        Slice the brain data into segments based on the window time stamps.
        Returns dictionary of brain segments for each frequency band, containing
        tensor of shape [B, C, T] where B (windows), C (channels), T (time steps).
        Each frequency band processed separately to reduce memory usage.

        Arguments:
            recording -- the recording object
            raw -- the raw data object
            brain_window_timestamps -- list of tuples of start and end time stamps
            batch_size -- number of windows to process at once
        """
        # Done first if many frequency bands to avoid duplication
        if self.notch_filter and recording.power_line_freq:
            recording.notch_filter(raw=raw, n_jobs=1)

        brain_segments = {}

        for band_name, (low, high) in self.frequency_bands.items():

            if self.frequency_bands.__len__() > 1:
                raw_band = raw.copy()
            else:
                raw_band = raw

            raw_band = recording.pre_process_brain(
                raw=raw_band,
                frequency_band=(low, high),
                scaling=self.scaling,
                brain_clipping=self.brain_clipping,
                baseline_window=self.baseline_window,
                new_freq=self.new_freq,
                n_jobs=self.n_jobs,
            )

            brain_tensor = torch.from_numpy(raw_band.get_data())  # [C, T]

            if cache:
                # Save frequency band specific tensor to cache, segmented when needed.
                torch.save(
                    brain_tensor,
                    f"{recording.cache_path}/{band_name}.pt",
                )
            del raw_band

            brain_segments[band_name] = self.segment_brain_tensor(
                brain_tensor=brain_tensor,
                recording=recording,
                brain_window_timestamps=brain_window_timestamps,
                batch_size=batch_size,
            )
            del brain_tensor

        return brain_segments

    def segment_brain_tensor(
        self,
        brain_tensor: torch.Tensor,
        recording: Recording,
        brain_window_timestamps: list[tuple[float, float]],
        batch_size: int,
    ):
        """Written separately to allow loading tensor from cache with time stamps.
        Slice the brain data into segments based on the window time stamps.

        Arguments:
            brain_tensor -- the brain tensor data of shape [C, T]
            recording -- the recording object
            brain_window_timestamps -- list of tuples of start and end time stamps
            batch_size -- number of windows to process at once

        Returns:
            torch.Tensor -- of shape [B, C, window_size]
        """
        # Initialize output arrays for each band, of shape [B, C, T] (with padding)
        window_size_points = self.window_size * self.new_freq
        n_windows = len(brain_window_timestamps)

        all_segments = torch.zeros(
            (n_windows, recording.channel_count, window_size_points),
            dtype=torch.float32,
        )

        start_indices = torch.tensor(
            [
                int((t[0] + recording.start_time + self.delay) * self.new_freq)
                for t in brain_window_timestamps
            ]
        )
        end_indices = start_indices + window_size_points

        # Extract windows in batches for memory efficiency
        for batch_start in range(0, n_windows, batch_size):

            batch_end = min(batch_start + batch_size, n_windows)
            batch_start_idx = start_indices[batch_start:batch_end]
            batch_end_idx = end_indices[batch_start:batch_end]

            # Extract all windows for this batch at once
            max_end_idx = brain_tensor.shape[1]

            for i, (start_idx, end_idx) in enumerate(
                zip(batch_start_idx, batch_end_idx)
            ):
                actual_end = min(end_idx.item(), max_end_idx)
                segment = brain_tensor[:, start_idx:actual_end]  # [C, T]

                # Exact dim match, add to all_segments
                if segment.shape[1] == window_size_points:
                    all_segments[batch_start + i] = segment
                # Truncate if too long, pad implicit from initialization
                else:
                    actual_size = segment.shape[1]
                    all_segments[batch_start + i, :, :actual_size] = segment
                del segment

        return all_segments
