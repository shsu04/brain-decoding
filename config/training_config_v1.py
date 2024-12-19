import typing as tp
from .simpleconv_config import SimpleConvConfig
from .config import Config
from .training_config import TrainingConfig

class TrainingConfigV1(TrainingConfig):
    """
    Config class for Whisper latent alignment, architecture exploration, 
    and dataset integration
    """
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
        alpha: float = 0.5,
        random_test_size: int = 3,
        seed: int = 42,
        # Training objective
        mel_alignment_objectives: dict[str, float] = {
            "clip_loss": 0.0,
            "mse_loss": 0.0,
        },
        latent_alignment_objectives: dict[str, float] = {
            "cosine_similarity": 0.0,
            "mse_loss": 0.0,
        },
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
        self.use_mse_loss = use_mse_loss
        self.alpha = alpha
        self.random_test_size = random_test_size
        self.seed = seed
        
        assert all([v >= 0 for v in mel_alignment_objectives.values()]), "Weighting must be non-negative"
        assert all([k in ["clip_loss", "mse_loss"] for k in mel_alignment_objectives.keys()]), "Invalid objective"
        self.mel_alignment_objectives = mel_alignment_objectives

        assert all([v >= 0 for v in latent_alignment_objectives.values()]), "Weighting must be non-negative"
        assert all([k in ["cosine_similarity", "mse_loss"] for k in latent_alignment_objectives.keys()]), "Invalid objective"
        self.latent_alignment_objectives = latent_alignment_objectives