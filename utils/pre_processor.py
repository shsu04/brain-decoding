from sklearn.preprocessing import RobustScaler, StandardScaler
import numpy as np
import librosa
import torch
import mne
import typing as tp


class PreProcessor:
    def __init__(
        self,
    ):
        """Pre-processes the brain and target data type.

        Keyword Arguments:
            brain_sample_rate -- new sample rate for the brain data (default: {100})

            audio_model -- model to use for audio processing (default: {"openai/whisper-large-v3"})
            audio_sample_rate -- sample rate for the audio data (default: {16000})
            audio_window_size -- window size for the audio data (default: {4})
            audio_hop_length -- hop length for the audio data (default: {160})
        """
        super(PreProcessor, self).__init__()
        self.INVALID = -0.1

    def pre_process_brain(
        self,
        raw: mne.io.Raw,
        channel_names: list[str],
        frequency_bands: dict = {"all": (0.5, 100)},
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
            channel_names -- list of channel names to be used for pre-processing
            frequency_bands -- dictionary of frequency bands to filter to (default: {"all": (0.5, 100)})
            brain_clipping -- standard deviation to clip the brain data to (default: {20})
            baseline_window -- window size for baseline correction in seconds (default: {0.5})
            new_freq -- new frequency to downsample the data to (default: {100})
            n_jobs -- number of jobs to run in parallel

        Returns:
            Dictionary of pre-processed raw data for each frequency band
        """
        results = {}

        robust_scaler, standard_scaler = RobustScaler(), StandardScaler()

        for band, (low, high) in frequency_bands.items():

            raw_copy = raw.copy() if frequency_bands.__len__() > 1 else raw

            # Band pass filter
            raw_copy = raw_copy.filter(
                picks=channel_names,
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
                    picks=channel_names,
                    channel_wise=True,
                    verbose=False,
                    n_jobs=n_jobs,
                )

            # Robust scaling
            raw_copy = raw_copy.apply_function(
                lambda x: robust_scaler.fit_transform(x.reshape(-1, 1)).ravel(),
                picks=channel_names,
                channel_wise=True,
                verbose=False,
                n_jobs=n_jobs,
            )

            # Standard scaling
            raw_copy = raw_copy.apply_function(
                lambda x: standard_scaler.fit_transform(x.reshape(-1, 1)).ravel(),
                picks=channel_names,
                channel_wise=True,
                verbose=False,
                n_jobs=n_jobs,
            )
            if brain_clipping:
                # Clipping
                raw_copy = raw_copy.apply_function(
                    lambda x: np.clip(
                        x, -brain_clipping, brain_clipping
                    ),  # Clip by multiples of standard deviations
                    picks=channel_names,
                    channel_wise=True,
                    verbose=False,
                    n_jobs=n_jobs,
                )

            results[band] = raw_copy

        return results

    def get_sensor_layout(self, raw: mne.io.Raw) -> torch.Tensor:
        """Returns the mix-max scaled sensor locations of the neural recording.

        Returns:
            positions of the sensors in 2D space. Dim = [C, 2]
        """
        info = raw.info
        layout = mne.find_layout(info)
        indexes: tp.List[int] = []
        valid_indexes: tp.List[int] = []

        # Separate valid from invalid channels
        for meg_index, name in enumerate(info.ch_names):
            name = name.rsplit("-", 1)[0]
            try:
                indexes.append(layout.names.index(name))
            except ValueError:
                print(
                    f"Channels {name} not in layout",
                )
            else:
                valid_indexes.append(meg_index)

        # [C, 2]
        positions = torch.full(size=(len(info.ch_names), 2), fill_value=self.INVALID)
        # The unit-normalized channel position in 2d (x, y, width, height)
        x, y = layout.pos[indexes, :2].T  # [2, C]

        # Scale again relative to valid channels
        x = (x - x.min()) / (x.max() - x.min())
        y = (y - y.min()) / (y.max() - y.min())

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        positions[valid_indexes, 0] = x
        positions[valid_indexes, 1] = y

        return positions

    def pre_process_audio(
        self,
        audio: np.ndarray,
        time_stamps: list[tuple[float, float]],
        audio_processor,
        window_size: int = 4,
        hop_length: int = 160,
        audio_sample_rate: int = 16000,
    ) -> dict[str, torch.Tensor]:
        """Pre-processes the audio data by segmenting it based on time stamps,
        returns Mel spectrogram features. Number of time steps will be
        window_size * sample_rate / hop_length.

        Arguments:
            audio -- audio data to be pre-processed: [T * sample_rate]

        Keyword Arguments:
            audio_processor -- Transforms the audio data into mel spectrogram
            time_stamps -- list of tuples containing the start and end time of segments
            window_size -- window size for the audio segments in seconds (default: {4})
            audio_sample_rate -- sample rate for the audio (default: {16000})
            hop_length -- hop length for the audio data in samples (default: {160})

        Returns:
            inputs -- pre-processed audio data, size [B, mel_bins, T]
        """
        audio_segments = []

        # Get the audio segments based on the time stamps and sample rate
        for start, end in time_stamps:
            start, end = int(start * audio_sample_rate), int(end * audio_sample_rate)
            audio_segments.append(audio[start:end])

        # Batch process the audio segments
        inputs = audio_processor(
            audio_segments,
            sampling_rate=audio_sample_rate,
            return_tensors="pt",
            do_normalize=True,
            hop_length=hop_length,
            max_length=audio_sample_rate * window_size,
        )

        return inputs["input_features"]
