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
