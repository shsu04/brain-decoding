"""
AudioBatch and its fetcher, used by DataLoader for parallel data loading.
Specifies the format and pre-processing steps to transform a study recording 
into batch of brain and audio segment pairs, sliced by window size and stride.
"""

import copy
import os
from attr import dataclass
import mne
import numpy as np
import pandas as pd
import ray
import torch
from transformers import WhisperFeatureExtractor

from studies import Study, Recording
from .batch import Batch, BatchFetcher


@dataclass
class AudioBatch(Batch):
    brain_segments: dict[str, torch.Tensor]
    audio_segments: torch.Tensor
    recording: Recording


@ray.remote
class AudioBatchFetcher(BatchFetcher):
    """Fetches a single recording from an audio and brain pair dataset."""

    def __init__(
        self,
        notch_filter: bool,
        frequency_bands: dict[str, tuple[float, float]],
        scaling: str,
        brain_clipping: int,
        baseline_window: float,
        new_freq: int,
        # Specific to this batch type
        max_random_shift: float,
        window_size: int,
        window_stride: int,
        audio_sample_rate: int,
        hop_length: int,
        audio_processor: str,
        n_jobs: int = 1,
    ):
        """
        Arguments:
            max_random_shift -- maximum random shift to apply to the windows
            window_size -- size of the window to extract
            audio_sample_rate -- sample rate for the audio data
            hop_length -- hop length for the audio data
            audio_processor -- model to use for audio processing

        Keyword Arguments:
            notch_filter -- whether to apply notch filter to the raw data to remove powerline
            frequency_bands -- dictionary of frequency bands tuple,
                brain segements will be returned for each band in the dictionary
            scaling -- scaling method to apply to the brain data
            brain_clipping -- standard deviation to clip the brain data to
            baseline_window -- window size to use for baseline normalization
            new_freq -- new frequency to resample the brain data to
        """
        assert audio_sample_rate // hop_length == new_freq

        self.notch_filter = notch_filter
        self.frequency_bands = frequency_bands
        self.scaling = scaling
        self.brain_clipping = brain_clipping
        self.baseline_window = baseline_window
        self.new_freq = new_freq

        # Specific to this batch type
        self.max_random_shift = max_random_shift
        self.window_size = window_size
        self.window_stride = window_stride
        self.audio_sample_rate = audio_sample_rate
        self.hop_length = hop_length
        self.audio_processor = WhisperFeatureExtractor.from_pretrained(audio_processor)
        self.n_jobs = n_jobs

    def fetch(self, recording: Recording, cache: bool) -> AudioBatch:
        """Load, pre-process, and slice audio and brain data into batch segments.
        Loads from cache if available. Audio is returned as mel spectrogram features
        of shape [B, mel_bins, T], brain data is returned as tensor of shape [B, C, T].

        cache -- whether to save the batch to disk for future use

        Raises:
            ValueError: Number of brain and audio windows do not match. Skip batch.
        """

        # Load from cache if available
        if os.path.exists(recording.cache_path):

            loaded = torch.load(recording.cache_path)
            return AudioBatch(
                brain_segments=loaded["brain"],
                audio_segments=loaded["audio"],
                recording=recording,
            )

        # Load the raw data
        try:
            raw = recording.load_raw(load_data=True)
            events = recording.load_events(raw=raw, options="sound")
            sound_events = events["sound"]

            # Generate time stamps for the windows
            audio_window_timestamps, brain_window_timestamps = (
                self._generate_time_stamps(sound_events)
            )

            # BRAIN
            brain_segments = self._segment_brain(
                recording=recording,
                raw=raw,
                brain_window_timestamps=brain_window_timestamps,
            )
            # AUDIO
            audio_segments = self._segment_audio(
                recording=recording,
                sound_events=sound_events,
                audio_window_timestamps=audio_window_timestamps,
            )

            # Brain audio dimensions check
            if (
                brain_segments[list(self.frequency_bands.keys())[0]].shape[0]
                != audio_segments.shape[0]
            ) or (
                brain_segments[list(self.frequency_bands.keys())[0]].shape[-1]
                != audio_segments.shape[-1]
            ):
                raise ValueError("Number of brain and audio windows do not match")

            # Cache the data
            if cache:
                torch.save(
                    {
                        "brain": brain_segments,
                        "audio": audio_segments,
                    },
                    recording.cache_path,
                )

            return AudioBatch(
                brain_segments=brain_segments,
                audio_segments=audio_segments,
                recording=recording,
            )

        except Exception as e:
            print(f"Error loading recording: {recording.cache_path}: {e}")
            return None

    def _generate_time_stamps(
        self, sound_events: pd.DataFrame
    ) -> tuple[dict[str, list[tuple[float, float]]], list[tuple[float, float]]]:
        """Obtain the list of start and end times for the windows. Making sure
        windows don't span two different sound files.

        Arguments:
            sound_events -- DataFrame containing the sound events. Columns are
                'onset', 'sound', 'end' onset is event marker in the brain data,
                start is the onset in the audio file

        Returns:
            audio_window_timestamps -- dictionary of list of tuples, list of tuples
            brain_window_timestamps -- list of tuples
        """
        audio_window_timestamps, brain_window_timestamps = {}, []

        for sound_file in sorted(sound_events["sound"].unique()):

            start_time, end_time = (
                sound_events[sound_events["sound"] == sound_file]["onset"].iloc[0],
                sound_events[sound_events["sound"] == sound_file]["end"].iloc[0],
            )

            audio_start_time = copy.deepcopy(start_time)
            audio_window_timestamps[sound_file] = []

            # This works on onset times (brain)
            while start_time + self.window_size < end_time:

                brain_window_timestamps.append(
                    (start_time, start_time + self.window_size)
                )

                # Notes the corresponsing timestamps in the audio file
                audio_window_timestamps[sound_file].append(
                    (
                        start_time - audio_start_time,
                        start_time + self.window_size - audio_start_time,
                    )
                )
                start_time += np.random.uniform(
                    self.window_stride, self.window_stride + self.max_random_shift
                )  # some randomness

        return audio_window_timestamps, brain_window_timestamps

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

    def _segment_audio(
        self,
        recording: Recording,
        sound_events: pd.DataFrame,
        audio_window_timestamps: dict[str, list[tuple[float, float]]],
    ) -> torch.Tensor:
        """Slice the audio data into segments based on the window time stamps.
        Pre-processes into Mel spectrogram features. Returns tensor of shape
        [B, mel_bins, T] where B (windows), mel_bins, T (time steps).
        """
        audio_segments = []

        for sound_file in sorted(sound_events["sound"].unique()):

            audio_segment = self._pre_process_audio(
                audio=recording.load_stimuli(sound_file),
                time_stamps=audio_window_timestamps[sound_file],
            )  # [B, mel_bins, T]
            audio_segment = audio_segment[
                :, :, : int(self.window_size * self.audio_sample_rate / self.hop_length)
            ]  # Truncate temporal dim
            audio_segments.append(audio_segment)

        # Concat along batch dim
        audio_segments = torch.cat(audio_segments, dim=0)

        return audio_segments

    def _pre_process_audio(
        self,
        audio: np.ndarray,
        time_stamps: list[tuple[float, float]],
    ) -> dict[str, torch.Tensor]:
        """Pre-processes the audio data by segmenting it based on time stamps,
        returns Mel spectrogram features. Number of time steps will be
        window_size * sample_rate / hop_length.

        Keyword Arguments:
            audio -- audio data to be pre-processed: [T * sample_rate]
            time_stamps -- list of tuples containing the start and end time of segments

        Returns:
            inputs -- pre-processed audio data, size [B, mel_bins, T]
        """
        audio_segments = []

        # Get the audio segments based on the time stamps and sample rate
        for start, end in time_stamps:
            start, end = int(start * self.audio_sample_rate), int(
                end * self.audio_sample_rate
            )
            audio_segments.append(audio[start:end])

        # Batch process the audio segments
        inputs = self.audio_processor(
            audio_segments,
            sampling_rate=self.audio_sample_rate,
            return_tensors="pt",
            do_normalize=True,
            hop_length=self.hop_length,
            max_length=self.audio_sample_rate * self.window_size,
        )

        return inputs["input_features"]
