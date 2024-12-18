"""
AudioBatch and its fetcher, used by DataLoader for parallel data loading.
Specifies the format and pre-processing steps to transform a study recording 
into batch of brain and audio segment pairs, sliced by window size and stride.
"""

import copy
import os
import pickle
import shutil
from attr import dataclass
import mne
import numpy as np
import pandas as pd
import ray
import torch
from transformers import WhisperFeatureExtractor
from typing import Dict, Tuple, Any
import torch
import shutil
from ray.exceptions import RayTaskError
import traceback

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
        delay: float,
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
            delay -- delay to apply to the brain data
        """
        assert audio_sample_rate // hop_length == new_freq

        self.notch_filter = notch_filter
        self.frequency_bands = frequency_bands
        self.scaling = scaling
        self.brain_clipping = brain_clipping
        self.baseline_window = baseline_window
        self.new_freq = new_freq
        self.n_jobs = n_jobs
        self.delay = delay

        # Specific to this batch type
        self.max_random_shift = max_random_shift
        self.window_size = window_size
        self.window_stride = window_stride
        self.audio_sample_rate = audio_sample_rate
        self.hop_length = hop_length
        self.audio_processor = WhisperFeatureExtractor.from_pretrained(audio_processor)

    def fetch(self, recording: Recording, cache: bool) -> AudioBatch:
        """
        Load, pre-process, and slice audio and brain data into batch segments.
        Loads from cache if available. Audio is returned as mel spectrogram features
        of shape [B, mel_bins, T], brain data is returned as tensor of shape [B, C, T].

        cache -- whether to save the batch to disk for future use

        Raises:
            ValueError: Number of brain and audio windows do not match. Skip batch.
        """
        # Initialize cache directory for the first time
        if not os.path.exists(recording.cache_path):
            os.makedirs(recording.cache_path)
        try:
            try:
                # Try loading from cache first
                (
                    brain_segments,
                    audio_window_timestamps,
                    brain_window_timestamps,
                    brain_start_time,
                    info,
                ) = self.fetch_cached_data(recording)

                recording.start_time = brain_start_time
                recording.info = info

                # Segment brain tensors
                for band in self.frequency_bands.keys():
                    brain_segments[band] = self.segment_brain_tensor(
                        brain_tensor=brain_segments[band],
                        recording=recording,
                        brain_window_timestamps=brain_window_timestamps,
                        batch_size=256,
                    )
            # Alternatively, process raw data
            except (ValueError, RayTaskError, FileNotFoundError) as e:
                brain_segments, audio_window_timestamps, brain_window_timestamps = (
                    self.fetch_raw_data(recording, cache=cache)
                )

            # AUDIO
            audio_segments = self.segment_audio(
                recording=recording,
                audio_window_timestamps=audio_window_timestamps,
            )
            # DIMENSION CHECKS
            # if not all of the brain segments are the same length
            if not all(
                [
                    brain_segments[list(self.frequency_bands.keys())[0]].shape[-1]
                    == brain_segments[band].shape[-1]
                    for band in self.frequency_bands.keys()
                ]
            ):
                raise ValueError(
                    f"Brain segments are not the same length: {recording.cache_path}"
                    + f" {brain_segments[list(self.frequency_bands.keys())[0]].shape[-1]}"
                )

            # B, and T mismatch between brain and audio
            if (
                brain_segments[list(self.frequency_bands.keys())[0]].shape[0]
                != audio_segments.shape[0]
            ) or (
                brain_segments[list(self.frequency_bands.keys())[0]].shape[-1]
                != audio_segments.shape[-1]
            ):
                raise ValueError("Number of brain and audio windows do not match")

            return AudioBatch(
                brain_segments=brain_segments,
                audio_segments=audio_segments,
                recording=recording,
            )

        except Exception as e:
            # Clean up cache if anything fails
            shutil.rmtree(recording.cache_path, ignore_errors=True)

            # Capture full traceback for Ray errors
            if isinstance(e, RayTaskError):
                error_msg = f"Ray task failed: {str(e)}\n{traceback.format_exc()}"
            else:
                error_msg = str(e)

            raise ValueError(
                f"Error fetching batch for {recording.cache_path}: {error_msg}"
            )

    def fetch_cached_data(
        self, recording: Recording
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, mne.Info]:
        """Loads unsliced brain tensor and timestamps from cache.
        Brain start time saved in case of recording does not start at 0,
        causing indexing issues when slicing brain tensor.
        """
        try:
            # IF cached
            brain_segments = {
                band: torch.load(f"{recording.cache_path}/{band}.pt")
                for band in self.frequency_bands.keys()
            }
            timestamps = torch.load(
                recording.cache_path + "/timestamps.pt"
            )  # Load timestamps

            info = pickle.load(open(recording.cache_path + "/info.pkl", "rb"))

            return (
                brain_segments,
                timestamps["audio_window_timestamps"],
                timestamps["brain_window_timestamps"],
                timestamps["brain_start_time"],
                info,
            )
        except Exception as e:
            raise ValueError(f"Cache loading failed: {str(e)}")

    def fetch_raw_data(
        self, recording: Recording, cache: bool
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Load raw, pre-process, and segment into tensors if not cached.
        Saves brain tensor and timestamps to cache for future use, including
        brain start time in case of recording does not start at 0.
        """
        raw = None
        try:
            # Clear cache
            shutil.rmtree(recording.cache_path, ignore_errors=True)
            os.makedirs(recording.cache_path)

            # IF not cached
            # Load the raw data
            raw = recording.load_raw(load_data=True)
            events = recording.load_events(raw=raw, options="sound")
            sound_events = events["sound"]

            # Generate time stamps for the windows
            audio_window_timestamps, brain_window_timestamps = (
                self.generate_time_stamps(sound_events)
            )

            # BRAIN
            brain_segments = self.segment_brain_mne(
                recording=recording,
                raw=raw,
                brain_window_timestamps=brain_window_timestamps,
                batch_size=256,  # Memory efficient batch size
                cache=cache,
            )

            if cache:
                torch.save(
                    {
                        "audio_window_timestamps": audio_window_timestamps,
                        "brain_window_timestamps": brain_window_timestamps,
                        "brain_start_time": recording.start_time,
                    },
                    recording.cache_path + "/timestamps.pt",
                )
                pickle.dump(
                    recording.info, open(recording.cache_path + "/info.pkl", "wb")
                )

            return brain_segments, audio_window_timestamps, brain_window_timestamps

        except Exception as e:
            raise ValueError(f"Raw data processing failed: {str(e)}")
        finally:
            if raw is not None:
                del raw

    def generate_time_stamps(
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

    def segment_audio(
        self,
        recording: Recording,
        audio_window_timestamps: dict[str, list[tuple[float, float]]],
    ) -> torch.Tensor:
        """Slice the audio data into segments based on the window time stamps.
        Pre-processes into Mel spectrogram features. Returns tensor of shape
        [B, mel_bins, T] where B (windows), mel_bins, T (time steps).
        """
        audio_segments = []

        # dict of audio_name: audio_data (np.array)
        audios = recording.load_stimuli(list(audio_window_timestamps.keys()))

        for sound_file in sorted(list(audio_window_timestamps.keys())):

            audio_segment = self.pre_process_audio(
                audio=audios[sound_file],
                time_stamps=audio_window_timestamps[sound_file],
            )  # [B, mel_bins, T]
            audio_segment = audio_segment[
                :,
                :,
                : int(self.window_size * self.audio_sample_rate / self.hop_length),
            ]  # Truncate temporal dim
            audio_segments.append(audio_segment)

        # Concat along batch dim
        audio_segments = torch.cat(audio_segments, dim=0)

        return audio_segments

    def pre_process_audio(
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

        time_stamps = torch.tensor(time_stamps)  # Shape: [N, 2]
        start_samples = (time_stamps[:, 0] * self.audio_sample_rate).to(torch.int64)
        end_samples = (time_stamps[:, 1] * self.audio_sample_rate).to(torch.int64)
        audio_segments = [
            audio[start:end] for start, end in zip(start_samples, end_samples)
        ]

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
