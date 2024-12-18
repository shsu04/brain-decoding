from abc import ABC, abstractmethod
import mne
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
import torch
from itertools import chain

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
        if not hasattr(self, "batch_type"):
            raise AttributeError("Study must have a batch type attribute.")

    @abstractmethod
    def create_recordings(self):
        """Creates Recording objects in self.recordings according to the study's directory structure"""
        pass

    def recording_exists(self, subject: str, session: str, task: str) -> bool:
        """Check if a recording exists for a given subject, session, and task."""
        try:
            exists = self.recordings[self.subjects.index(subject)][
                self.sessions.index(session)
            ][self.tasks.index(task)]

        except Exception as e:
            exists = False
        return exists

    def get_flat_recordings_list(self):
        """Collapses 3D recordings list into single list of recordings."""
        return list(chain.from_iterable(chain.from_iterable(self.recordings)))


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
        self.channel_count = len(channel_names)
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
    def load_stimuli(self, names: list[str], options: str) -> dict[str, np.ndarray]:
        """Each study has their own stimuli, e.g. audio, visual, etc."""
        pass

    def notch_filter(self, raw: mne.io.Raw, n_jobs=1):
        """
        Applies notch filter to remove powerline noise and its harmonices, in Hz.
        Done here to avoid multiple raw copies and pre_processing steps for each
        frequency band filters.
        """
        if not self.power_line_freq:
            print("Power line frequency not set, skipping notch filter.")
            return raw

        return raw.notch_filter(
            freqs=[self.power_line_freq * i for i in range(1, 10)],
            verbose=False,
            n_jobs=n_jobs,
        )

    def pre_process_brain(
        self,
        raw: mne.io.Raw,
        frequency_band: tuple[float, float] = (0.5, 100),
        scaling: str = "both",
        brain_clipping: int = 20,
        baseline_window: float = 0.5,
        new_freq: int = 100,
        n_jobs: int = None,
    ) -> dict[str, mne.io.Raw]:
        """
        Pre-processes the raw brain data by applying band-pass filter, down-sampling,
        baseline correction, robust scaling, standard scaling, and clamping. Function
        to be called separately for each frequency band to avoid multiple raw copies,
        hence notch filter should be applied before calling this function.

        Keyword Arguments:
            raw -- mne.io.Raw object of the session to be pre-processed
            frequency_bands -- dict of freq bands to filter to. Default is (0.5, 100)
            scaling -- type of scaling to apply, can be "robust", "standard", or "both"
            brain_clipping -- standard deviation to clip the brain data to (default: {20})
            baseline_window -- window size for baseline correction in seconds
            new_freq -- new frequency to downsample the data to (default: {100})
            n_jobs -- number of jobs to run in parallel

        Returns:
            mne.io.Raw object of the pre-processed brain data
        """
        low, high = frequency_band

        assert scaling in [
            "robust",
            "standard",
            "both",
        ], f"Invalid scaling type {scaling}"
        if baseline_window:
            assert baseline_window >= 0
        if low is not None and high is not None:
            assert low < high, "Low frequency must be less than high frequency."

        robust_scaler = RobustScaler() if scaling in ["robust", "both"] else None
        standard_scaler = StandardScaler() if scaling in ["standard", "both"] else None

        if not hasattr(self, "start_time"):
            self.start_time = raw.times[0]

        # Band pass filter
        raw = raw.filter(
            picks=self.channel_names,
            l_freq=low,
            h_freq=high,
            verbose=False,
            n_jobs=n_jobs,
        )

        # Downsample
        raw = raw.resample(sfreq=new_freq, verbose=False, n_jobs=n_jobs)

        if baseline_window:
            # Baseline correction by first 0.5 secs
            raw = raw.apply_function(
                lambda x: x - np.mean(x[: int(baseline_window * new_freq)]),
                picks=self.channel_names,
                channel_wise=True,
                verbose=False,
                n_jobs=n_jobs,
            )

        # Robust scaling
        if robust_scaler:
            raw = raw.apply_function(
                lambda x: robust_scaler.fit_transform(x.reshape(-1, 1)).ravel(),
                picks=self.channel_names,
                channel_wise=True,
                verbose=False,
                n_jobs=n_jobs,
            )

        if brain_clipping:
            # First calculate channel-wise standard deviation
            def clip_by_std(x):
                std = np.std(x)
                return np.clip(x, -brain_clipping * std, brain_clipping * std)

            raw = raw.apply_function(
                clip_by_std,
                picks=self.channel_names,
                channel_wise=True,
                verbose=False,
                n_jobs=n_jobs,
            )

        # Standard scaling
        if standard_scaler:
            raw = raw.apply_function(
                lambda x: standard_scaler.fit_transform(x.reshape(-1, 1)).ravel(),
                picks=self.channel_names,
                channel_wise=True,
                verbose=False,
                n_jobs=n_jobs,
            )

        return raw
