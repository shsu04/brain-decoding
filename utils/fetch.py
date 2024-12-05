from email.mime import audio
from studies import Study
from .pre_processor import PreProcessor
import torch
import copy
import numpy as np
import os


def fetch_audio_and_brain_pairs(
    subject: int,
    task: int,
    session: int,
    max_random_shift: float,
    window_size: int,
    window_stride: int,
    study: Study,
    pre_processor: PreProcessor,
    frequency_bands: dict = {"all": (0.5, 100)},
    brain_clipping: int = 20,
    audio_sample_rate: int = 16000,
    hop_length: int = 160,
    n_jobs: int = -1,
) -> tuple[dict[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Fetches and pre-processes audio and brain data for a given subject,
    task, and session.

    Arguments:
        max_random_shift -- maximum random shift to apply to the windows
        window_size -- size of the window to extract

    Keyword Arguments:
        frequency_bands -- dictionary of frequency bands tuple,
            brain segements will be returned for each band in the dictionary
        brain_clipping -- standard deviation to clip the brain data to
        audio_sample_rate -- sample rate for the audio data
        hop_length -- hop length for the audio data

    Raises:
        ValueError: Number of brain and audio windows do not match. Skip batch.

    Returns:
        brain_segments -- dictionary of brain segments for each frequency band [B, C, T]
        audio_segments -- audio segments for the session [B, mel_bins, T]
        layout -- sensor layout for the session [C, 2]
    """
    assert audio_sample_rate // hop_length == pre_processor.new_freq

    # See if cached
    if os.path.exists(f"{study.cache_dir}/{subject}_{task}_{session}.pt"):
        loaded = torch.load(f"{study.cache_dir}/{subject}_{task}_{session}.pt")
        return loaded["brain"], loaded["audio"], loaded["layout"]

    # Get attributes for the recording
    raw, word_events, sound_events = study.clean_recording(
        subject, task, session, n_jobs=n_jobs
    )
    layout = pre_processor.get_sensor_layout(raw)  # [C, 2]

    # TIMESTAMP
    # Obtain the list of start and end times for the windows
    # Making sure windows don't span two different sound files
    # dict of list of tuples, list of tuples
    audio_window_timestamps, brain_window_timestamps = {}, []

    for sound_file in sorted(sound_events["sound"].unique()):

        start_time, end_time = (
            sound_events[sound_events["sound"] == sound_file]["onset"].iloc[0],
            sound_events[sound_events["sound"] == sound_file]["end"].iloc[0],
        )

        audio_start_time = copy.deepcopy(start_time)
        audio_window_timestamps[sound_file] = []

        # This works on onset times (brain)
        while start_time + window_size < end_time:

            brain_window_timestamps.append((start_time, start_time + window_size))

            # Notes the corresponsing timestamps in the audio file
            audio_window_timestamps[sound_file].append(
                (
                    start_time - audio_start_time,
                    start_time + window_size - audio_start_time,
                )
            )
            start_time += np.random.uniform(window_stride, window_stride + max_random_shift)  # some randomness

    # BRAIN
    results = pre_processor.pre_process_brain(
        raw,
        channel_names=study.channel_names,
        n_jobs=n_jobs,
        frequency_bands=frequency_bands,
        brain_clipping=brain_clipping,
    )
    # np array of time stamps corresponsing to brain data
    times = torch.from_numpy(results[list(frequency_bands.keys())[0]].times)

    n_windows = len(brain_window_timestamps)
    data, brain_segments = {}, {}

    # Initialize output arrays for each band, of shape [B, C, T] (with padding)
    for band in frequency_bands.keys():

        data[band] = torch.from_numpy(results[band].get_data())  # [C, T]
        n_channels = data[band].shape[0]
        brain_segments[band] = torch.zeros(
            (n_windows, n_channels, window_size * pre_processor.new_freq),
            dtype=torch.float32,
        )

    # Extract windows for all bands
    for i, (start_time, end_time) in enumerate(brain_window_timestamps):

        mask = (times >= start_time) & (times <= end_time)

        for band in frequency_bands.keys():
            window = data[band][:, mask]  # Extract window
            brain_segments[band][i] = window[
                :, : window_size * pre_processor.new_freq
            ]  # Truncate to [C, T]

    # AUDIO
    audio_segments = []

    for sound_file in sorted(sound_events["sound"].unique()):

        audio_segment = pre_processor.pre_process_audio(
            path=os.path.join(study.root_dir, sound_file),
            window_size=window_size,
            hop_length=hop_length,
            time_stamps=audio_window_timestamps[sound_file],
        )  # [B, mel_bins, T]
        audio_segment = audio_segment[
            :, :, : int(window_size * audio_sample_rate / hop_length)
        ]  # Truncate temporal dim
        audio_segments.append(audio_segment)

    audio_segments = torch.cat(audio_segments, dim=0)

    # Brain audio dimensions check
    if (
        brain_segments[list(frequency_bands.keys())[0]].shape[0]
        != audio_segments.shape[0]
    ) or (
        brain_segments[list(frequency_bands.keys())[0]].shape[-1]
        != audio_segments.shape[-1]
    ):
        raise ValueError("Number of brain and audio windows do not match")

    # Cache the data
    torch.save(
        {
            "brain": brain_segments,
            "audio": audio_segments,
            "layout": layout,
        },
        f"{study.cache_dir}/{subject}_{task}_{session}.pt",
    )

    return brain_segments, audio_segments, layout
