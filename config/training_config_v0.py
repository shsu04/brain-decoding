import typing as tp
from .simpleconv_config import SimpleConvConfig
from .config import Config
from .training_config import TrainingConfig

class TrainingConfigV0(TrainingConfig):
    """Config class for pre-processing exploration with CLIP MSE loss."""
    def __init__(
        self,
        brain_encoder_config: SimpleConvConfig,
        data_partition: tp.Dict[str, tp.Dict[str, tp.List[str]]],
        # Pre-processing parameters
        # Brain
        new_freq: int = 100,
        frequency_bands: tp.Dict[str, tp.Tuple[float, float]] = {"all": (0.5, 100)},
        max_random_shift: float = 2.0,
        window_size: int = 4,
        window_stride: int = 1,
        brain_clipping: float = 20,
        baseline_window: int = 0.5,
        notch_filter: bool = True,
        scaling: str = "minmax",
        delay: float = 0,
        # Audio
        audio_model: str = "openai/whisper-large-v3",
        audio_sample_rate: int = 16000,
        hop_length: int = 160,
        # Hyperparameters
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        epochs: int = 50,
        batch_size: int = 128,
        use_clip_loss: bool = True,
        use_mse_loss: bool = True,
        alpha: float = 0.5,
        random_test_size: int = 3,
        seed: int = 42,
    ):
        self.brain_encoder_config = brain_encoder_config
        # key: study_name, value: dict with keys: "testing_subjects", "testing_tasks",
        # where each value is a list of int. Ones not specified in either lists are
        # used for training.
        self.data_partition = data_partition

        # Pre-processing parameters
        # Brain
        self.new_freq = new_freq
        self.frequency_bands = frequency_bands
        self.max_random_shift = max_random_shift
        self.window_size = window_size
        self.window_stride = window_stride
        self.baseline_window = baseline_window
        self.notch_filter = notch_filter
        self.brain_clipping = brain_clipping
        self.scaling = scaling
        self.delay = delay

        # Audio
        self.audio_model = audio_model
        self.audio_sample_rate = audio_sample_rate
        self.hop_length = hop_length

        # Hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_clip_loss = use_clip_loss
        self.use_mse_loss = use_mse_loss
        self.alpha = alpha
        self.random_test_size = random_test_size
        self.seed = seed

        assert 0 <= self.alpha <= 1, "Alpha must be between 0 and 1"
        assert use_clip_loss or use_mse_loss, "At least one loss function must be used"

    # does not overide parent method
    def to_dict(self):
        brain_encoder_config = self.brain_encoder_config.to_dict()
        config = super().to_dict()
        config["brain_encoder_config"] = brain_encoder_config
        return config

    def from_dict(self, config: tp.Dict[str, tp.Any]):
        self.brain_encoder_config = SimpleConvConfig.from_dict(
            config["brain_encoder_config"]
        )
        self.data_partition = config["data_partition"]
        self.new_freq = config["new_freq"]
        self.frequency_bands = config["frequency_bands"]
        self.max_random_shift = config["max_random_shift"]
        self.window_size = config["window_size"]
        self.window_stride = config["window_stride"]
        self.baseline_window = config["baseline_window"]
        self.notch_filter = config["notch_filter"]
        self.brain_clipping = config["brain_clipping"]
        self.scaling = config["scaling"]
        self.delay = config["delay"]
        self.audio_model = config["audio_model"]
        self.audio_sample_rate = config["audio_sample_rate"]
        self.hop_length = config["hop_length"]
        self.learning_rate = config["learning_rate"]
        self.weight_decay = config["weight_decay"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.use_clip_loss = config["use_clip_loss"]
        self.use_mse_loss = config["use_mse_loss"]
        self.alpha = config["alpha"]
        self.random_test_size = config["random_test_size"]
        self.seed = config["seed"]
        return self
