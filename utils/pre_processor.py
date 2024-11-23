from sklearn.preprocessing import RobustScaler, StandardScaler
import numpy as np
from transformers import AutoProcessor
import librosa
import torch
import matplotlib.pyplot as plt
import mne


class PreProcessor:
    def __init__(
        self,
        brain_sample_rate=100,
        frequency_bands: dict = {"all": (0.5, 100)},
        brain_clipping: float = 20.0,
        audio_model: str = "openai/whisper-large-v3",
        audio_sample_rate: int = 16000,
        audio_window_size: int = 4,
        audio_hop_length: int = 160,
    ):
        """Pre-processes the brain and target data type.

        Keyword Arguments:
            brain_sample_rate -- new sample rate for the brain data (default: {100})
            frequency_bands -- dictionary of frequency bands to filter to (default: {"all": (0.5, 100)})
            brain_clipping -- standard deviation to clip the brain data to (default: {20})

            audio_model -- model to use for audio processing (default: {"openai/whisper-large-v3"})
            audio_sample_rate -- sample rate for the audio data (default: {16000})
            audio_window_size -- window size for the audio data (default: {4})
            audio_hop_length -- hop length for the audio data (default: {160})
        """
        super(PreProcessor, self).__init__()
        # Brain
        self.new_freq = brain_sample_rate
        self.frequency_bands = frequency_bands
        self.brain_clipping = brain_clipping

        # Audio
        self.audio_model = audio_model
        self.audio_sample_rate = audio_sample_rate
        self.audio_window_size = audio_window_size
        self.audio_hop_length = audio_hop_length

    def pre_process_brain(
        self,
        raw: mne.io.Raw,
        data_type: str = "meg",
        n_jobs: int = None,
    ) -> dict[str, mne.io.Raw]:
        """
        Pre-processes the raw brain data by applying band-pass filter, down-sampling,
        baseline correction, robust scaling, standard scaling, and clamping.

        Keyword Arguments:
            raw -- mne.io.Raw object of the session to be pre-processed
            data_type -- type of data to be pre-processed (default: {"meg"})
            n_jobs -- number of jobs to run in parallel

        Returns:
            Dictionary of pre-processed raw data for each frequency band
        """
        results = {}

        robust_scaler, standard_scaler = RobustScaler(), StandardScaler()

        for band, (low, high) in self.frequency_bands.items():

            raw_copy = raw.copy() if self.frequency_bands.__len__() > 1 else raw

            # Band pass filter 0.5-100 Hz
            raw_copy = raw_copy.filter(
                picks=[data_type],
                l_freq=low,
                h_freq=high,
                verbose=False,
                n_jobs=n_jobs,
            )

            # Downsample
            raw_copy = raw_copy.resample(
                sfreq=self.new_freq, verbose=False, n_jobs=n_jobs
            )

            # Baseline correction  by first 0.5 secs
            raw_copy = raw_copy.apply_function(
                lambda x: x - np.mean(x[: int(0.5 * self.new_freq)]),
                picks=[data_type],
                channel_wise=True,
                verbose=False,
                n_jobs=n_jobs,
            )

            # Robust scaling
            raw_copy = raw_copy.apply_function(
                lambda x: robust_scaler.fit_transform(x.reshape(-1, 1)).ravel(),
                picks=[data_type],
                channel_wise=True,
                verbose=False,
                n_jobs=n_jobs,
            )

            # Standard scaling
            raw_copy = raw_copy.apply_function(
                lambda x: standard_scaler.fit_transform(x.reshape(-1, 1)).ravel(),
                picks=[data_type],
                channel_wise=True,
                verbose=False,
                n_jobs=n_jobs,
            )

            # Clamping
            raw_copy = raw_copy.apply_function(
                lambda x: np.clip(
                    x, -self.brain_clipping, self.brain_clipping
                ),  # Clip by multiples of standard deviations
                picks=[data_type],
                channel_wise=True,
                verbose=False,
                n_jobs=n_jobs,
            )

            results[band] = raw_copy

        return results

    def pre_process_audio(
        self,
        path: str,
        time_stamps: list[tuple[float, float]] = [(0.5, 1.5)],
    ) -> dict[str, torch.Tensor]:
        """Pre-processes the audio data by loading the audio file, segmenting
        it based on time stamps, returns Mel spectrogram features.

        Arguments:
            path -- path to the audio file

        Keyword Arguments:
            time_stamps -- list of tuples containing the start and end time
            stamps (default: {[(0.5, 1.5)]})

        Returns:
            inputs -- dictionary containing the pre-processed audio data
        """
        # Only initialize the processor if this function is called for the first time
        if not hasattr(self, "audio_processor"):
            self.audio_processor = AutoProcessor.from_pretrained(self.audio_model)

        audio, _ = librosa.load(path, sr=self.audio_sample_rate)
        audio = audio.astype(np.float32)

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
            hop_length=self.audio_hop_length,
            max_length=self.audio_sample_rate * self.audio_window_size,
        )

        return inputs

    def show_mel_spectrogram(
        self,
        x: torch.Tensor,
        max_plots: int = 4,
        x_pred: torch.Tensor = None,
        frequency_scale: str = "mel",  # Add parameter to choose between 'mel' and 'hz'
    ):
        """Plot log-mel spectrogram of the audio data.
        Plot side by side the original and predicted audio data if provided.

        Arguments:
            x -- audio data tensor of shape [B, mel_bins, T]
            max_plots -- maximum number of plots to display (default: {4})
            x_pred -- predicted audio data tensor of shape [B, mel_bins, T] (default: {None})
            frequency_scale -- y-axis scale ('mel' or 'hz') (default: {'mel'})
        """
        # Convert tensor to numpy array
        if torch.is_tensor(x):
            specs = x.detach().cpu().numpy()

        if x_pred is not None and torch.is_tensor(x_pred):
            specs_pred = x_pred.detach().cpu().numpy()

        num_specs = min(max_plots, specs.shape[0])
        cols = 2 if x_pred is not None else 1

        fig, axes = plt.subplots(num_specs, cols, figsize=(7 * cols, 2 * num_specs))

        for i in range(num_specs):
            if num_specs == 1:
                ax = axes if cols == 1 else axes[0]
                ax_pred = None if cols == 1 else axes[1]
            else:
                ax = axes[i, 0] if cols == 2 else axes[i]
                ax_pred = None if cols == 1 else axes[i, 1]

            # Plot original spectrogram
            img = librosa.display.specshow(
                specs[i],
                sr=self.audio_sample_rate,
                hop_length=self.audio_hop_length,
                x_axis="time",
                y_axis=frequency_scale,  # 'mel' or 'hz'
                ax=ax,
                cmap="viridis",
            )
            plt.colorbar(img, ax=ax, format="%+2.0f dB")
            y_label = (
                "Frequency (Hz)" if frequency_scale == "hz" else "Mel Frequency Bins"
            )
            ax.set_ylabel(y_label)
            ax.set_title(f"Original Log-Mel Spectrogram {i+1}")

            # Plot predicted spectrogram if available
            if x_pred is not None:
                img_pred = librosa.display.specshow(
                    specs_pred[i],
                    sr=self.audio_sample_rate,
                    hop_length=self.audio_hop_length,
                    x_axis="time",
                    y_axis=frequency_scale,  # 'mel' or 'hz'
                    ax=ax_pred,
                    cmap="viridis",
                )
                plt.colorbar(img_pred, ax=ax_pred, format="%+2.0f dB")
                ax_pred.set_ylabel(y_label)
                ax_pred.set_title(f"Predicted Log-Mel Spectrogram {i+1}")

        plt.tight_layout()
        plt.show()
