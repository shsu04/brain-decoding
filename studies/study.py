from abc import ABC, abstractmethod

import mne
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
import torch

from .stimuli import Stimuli


class Study(ABC):
    """
    Abstract class for a study. Mainly to create recording objects for a given
    study, and to specify study types. Its recordings are partitioned and fetched
    from the training loop to avoid deadlock when fetching recordings.
    """

    def __init__(self):
        super(Study, self).__init__()

        if not hasattr(self, "root_dir"):
            raise AttributeError("Study must have a root_dir attribute.")
        if not hasattr(self, "cache_dir"):
            raise AttributeError("Study must have a cache_dir attribute.")
        if not hasattr(self, "subjects_info"):
            raise AttributeError("Study must have a subjects_info attribute.")
        if not hasattr(self, "subjects"):
            raise AttributeError("Study must have a subjects_list attribute.")
        if not hasattr(self, "sessions"):
            raise AttributeError("Study must have a sessions attribute.")
        if not hasattr(self, "tasks"):
            raise AttributeError("Study must have a tasks attribute.")
        if not hasattr(self, "recordings"):
            raise AttributeError("Study must have a recordings attribute.")
        if not hasattr(self, "stimuli"):
            raise AttributeError("Study must have a stimuli attribute.")
        if not hasattr(self, "source_link"):
            raise AttributeError("Study must have a source_link attribute.")
        if not hasattr(self, "channel_names"):
            raise AttributeError(
                "Study must have channel names for filtering and sensor position"
            )
        if not hasattr(self, "recordings"):
            raise AttributeError("Study must have recordings attribute.")
        if not hasattr(self, "stimuli"):
            raise AttributeError("Study must have stimuli attribute.")


class Recording(ABC):
    """
    Abstract class for a recording. Mainly to create recording objects for a given
    study. Each recording must have the bids path, cache path, study, subject, session,
    relevant channel_names, info, and stumuli class.

    Each class must implement the load_raw and load_events and load_stimuli methods.
    """

    def __init__(
        self,
        bids_path: str,
        cache_path: str,
        study_name: str,
        subject_id: str,
        session_id: str,
        task_id: str,
        channel_names: list[str],
        stimuli: Stimuli,
        type: str,
        power_line_freq: int,
    ):
        super(Recording, self).__init__()

        self.bids_path = bids_path
        self.cache_path = cache_path
        self.study_name = study_name
        self.subject_id = subject_id
        self.session_id = session_id
        self.task_id = task_id
        self.channel_names = channel_names
        self.stimuli = stimuli
        self.type = type
        self.power_line_freq = power_line_freq
        self.info = None  # added when the recording is loaded.

    @abstractmethod
    def load_raw(self) -> mne.io.Raw:
        """Loads the raw data fwith only the pre-defined relevant channels."""
        pass

    @abstractmethod
    def load_events(self, raw: mne.io.Raw, options: str) -> dict[str, pd.DataFrame]:
        """Some studies have more than 1 type of events."""
        pass

    @abstractmethod
    def load_stimuli(self, name: str, options: str) -> dict[str, np.ndarray]:
        """Each study has their own stimuli, e.g. audio, visual, etc."""
        pass

    def load_sensor_layout(
        self, dim: int = None, proj: bool = False, scaling: str = "midpoint"
    ) -> torch.Tensor:
        """
        Returns the scaled sensor locations of the neural recording.
        Channels already valid from since picked in load raw

        If proj, 3D layout is projected to 2D space.

        Arguments:
            scaling -- type of scaling to apply, can be:
                "midpoint" - scale to [-1, 1] around geometric midpoint
                "minmax" - scale to [0, 1] based on min and max
                "standard" - scale to mean 0 and std 1
                "maxabs" - scale to [-1, 1] based on max absolute value

        Returns:
            if dim == 2:
                scaled positions of the sensors in 2D space. Dim = [C, 2], (x, y)
            if dim == 3:
                scaled positions of the sensors in 3D space. Dim = [C, 3], (x, y, z)
        """
        assert dim in [2, 3, None], f"Layout can only be 2D or 3D. Invalid dim {dim}."

        if not hasattr(self, "info"):
            self.load_raw()

        # Find layout projects 3D to 2D
        if proj:
            layout = mne.find_layout(
                self.info,
            )
            layout = torch.tensor(
                layout.pos[: len(self.channel_names), :2], dtype=torch.float32
            )  # [C, 2]
        # Get x, y, z coordinates without projection
        else:
            layout = torch.tensor(
                [
                    self.info["chs"][i]["loc"][:dim]
                    for i in range(len(self.info["chs"]))
                    if self.info["ch_names"][i] in self.channel_names
                ],
                dtype=torch.float32,
            )  # [C, dim]

        # Scaling each dim independently
        if scaling == "midpoint":
            midpoints = layout.mean(dim=0)
            max_deviation = (layout - midpoints).abs().max(dim=0).values
            layout = (layout - midpoints) / max_deviation

        elif scaling == "minmax":
            mins = layout.min(dim=0).values
            maxs = layout.max(dim=0).values
            layout = (layout - mins) / (maxs - mins)

        elif scaling == "standard":
            means = layout.mean(dim=0)
            stds = layout.std(dim=0)
            layout = (layout - means) / stds

        elif scaling == "maxabs":
            max_abs = layout.abs().max(dim=0).values
            layout = layout / max_abs

        elif scaling is not None:
            raise ValueError(f"Unsupported scaling type: {scaling}")

        return layout

    def pre_process_brain(
        self,
        raw: mne.io.Raw,
        notch_filter: bool = False,
        frequency_bands: dict = {"all": (0.5, 100)},
        scaling: str = "both",
        brain_clipping: int = 20,
        baseline_window: float = 0.5,
        new_freq: int = 100,
        n_jobs: int = None,
    ) -> dict[str, mne.io.Raw]:
        """
        Pre-processes the raw brain data by applying band-pass filter, down-sampling,
        baseline correction, robust scaling, standard scaling, and clamping.

        Keyword Arguments:
            raw -- mne.io.Raw object of the session to be pre-processed
            notch_filter -- removes the powerline noise and its harmonics, value in Hz
            frequency_bands -- dict of freq bands to filter to, e.g. {"all": (0.5, 100)}
            scaling -- type of scaling to apply, can be "robust", "standard", or "both"
            brain_clipping -- standard deviation to clip the brain data to (default: {20})
            baseline_window -- window size for baseline correction in seconds
            new_freq -- new frequency to downsample the data to (default: {100})
            n_jobs -- number of jobs to run in parallel

        Returns:
            Dictionary of pre-processed raw data for each frequency band
        """
        results = {}

        assert scaling in [
            "robust",
            "standard",
            "both",
        ], f"Invalid scaling type {scaling}"
        assert baseline_window >= 0

        robust_scaler = RobustScaler() if scaling in ["robust", "both"] else None
        standard_scaler = StandardScaler() if scaling in ["standard", "both"] else None

        # Determined by visual inspection of the data, exclude powerline noise.
        if notch_filter:
            if not self.power_line_freq:
                print("Power line frequency not set, skipping notch filter.")
            else:
                raw = raw.notch_filter(
                    freqs=[self.power_line_freq * i for i in range(1, 10)],
                    verbose=False,
                    n_jobs=n_jobs,
                )

        for band, (low, high) in frequency_bands.items():

            raw_copy = raw.copy() if frequency_bands.__len__() > 1 else raw

            # Band pass filter
            raw_copy = raw_copy.filter(
                picks=self.channel_names,
                l_freq=low,
                h_freq=high,
                verbose=False,
                n_jobs=n_jobs,
            )

            # Downsample
            raw_copy = raw_copy.resample(sfreq=new_freq, verbose=False, n_jobs=n_jobs)

            if baseline_window:
                # Baseline correction by first 0.5 secs
                raw_copy = raw_copy.apply_function(
                    lambda x: x - np.mean(x[: int(baseline_window * new_freq)]),
                    picks=self.channel_names,
                    channel_wise=True,
                    verbose=False,
                    n_jobs=n_jobs,
                )

            # Robust scaling
            if robust_scaler:
                raw_copy = raw_copy.apply_function(
                    lambda x: robust_scaler.fit_transform(x.reshape(-1, 1)).ravel(),
                    picks=self.channel_names,
                    channel_wise=True,
                    verbose=False,
                    n_jobs=n_jobs,
                )

            # Standard scaling
            if standard_scaler:
                raw_copy = raw_copy.apply_function(
                    lambda x: standard_scaler.fit_transform(x.reshape(-1, 1)).ravel(),
                    picks=self.channel_names,
                    channel_wise=True,
                    verbose=False,
                    n_jobs=n_jobs,
                )

            # Clipping
            if brain_clipping:
                raw_copy = raw_copy.apply_function(
                    lambda x: np.clip(
                        x, -brain_clipping, brain_clipping
                    ),  # Clip by multiples of standard deviations
                    picks=self.channel_names,
                    channel_wise=True,
                    verbose=False,
                    n_jobs=n_jobs,
                )

            results[band] = raw_copy

        return results
