import typing as tp
from .simpleconv_config import SimpleConvConfig
from .spectralconv_config import SpectralConvConfig
from .training_config import TrainingConfig

from peft import AdaLoraConfig


class TrainingConfigV2(TrainingConfig):
    """
    Config class for Whisper decoding and combining datasets.
    """

    def __init__(
        self,
        brain_encoder_config: tp.Union[SimpleConvConfig, SpectralConvConfig],
        data_partition: tp.Dict[str, tp.Dict[str, tp.List[str]]],
        use_adalora: bool = True,
        adalora_init_r: int = 8,
        adalora_target_r: int = 4,
        adalora_tinit: int = 500,
        adalora_tfinal: int = 0,
        adalora_deltaT: int = 100,
        adalora_lora_alpha: int = 32,
        adalora_lora_dropout: float = 0.1,
        adalora_total_step: int = None,
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
        scaling: str = "both",
        delay: float = 0,
        # Audio
        audio_model: str = "openai/whisper-tiny.en",
        audio_sample_rate: int = 16000,
        hop_length: int = 160,
        # Hyperparameters
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        epochs: int = 50,
        steps_per_epoch: int = 400,
        batch_size: int = 128,
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
            "clip_loss": 0.0,
            "mmd_loss": 0.0,
        },
        # Decoding params
        decode_timestamps: bool = True,
    ):
        self.brain_encoder_config = brain_encoder_config
        # key: study_name, value: dict with keys: "testing_subjects", "testing_tasks",
        # where each value is a list of int. Ones not specified in either lists are
        # used for training.
        self.data_partition = data_partition
        
        self.use_adalora = use_adalora
        if use_adalora:
            self.adalora_config = AdaLoraConfig(
                peft_type="ADALORA",
                task_type="SPEECH_RECOGNITION",
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
                init_r=adalora_init_r,
                target_r=adalora_target_r,
                tinit=adalora_tinit,
                tfinal=adalora_tfinal,
                deltaT=adalora_deltaT,
                lora_alpha=adalora_lora_alpha,
                lora_dropout=adalora_lora_dropout,
                total_step=adalora_total_step,
            )

            self.adalora_init_r = adalora_init_r
            self.adalora_target_r = adalora_target_r
            self.adalora_tinit = adalora_tinit
            self.adalora_tfinal = adalora_tfinal
            self.adalora_deltaT = adalora_deltaT
            self.adalora_lora_alpha = adalora_lora_alpha
            self.adalora_lora_dropout = adalora_lora_dropout
            self.adalora_total_step = adalora_total_step
        else:
            self.adalora_config = None
            self.adalora_init_r = None
            self.adalora_target_r = None
            self.adalora_tinit = None
            self.adalora_tfinal = None
            self.adalora_deltaT = None
            self.adalora_lora_alpha = None
            self.adalora_lora_dropout = None
            self.adalora_total_step = None

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
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_test_size = random_test_size
        self.seed = seed
        self.steps_per_epoch = steps_per_epoch

        # MEL ALIGNMENT
        assert all(
            [v >= 0 for v in mel_alignment_objectives.values()]
        ), "Weighting must be non-negative"
        assert all(
            [k in ["clip_loss", "mse_loss"] for k in mel_alignment_objectives.keys()]
        ), "Invalid objective"
        self.mel_alignment_objectives = mel_alignment_objectives

        # LATENT ALIGNMENT
        assert all(
            [v >= 0 for v in latent_alignment_objectives.values()]
        ), "Weighting must be non-negative"
        assert all(
            [
                k in ["cosine_similarity", "mse_loss", "clip_loss", "mmd_loss"]
                for k in latent_alignment_objectives.keys()
            ]
        ), "Invalid objective"

        self.latent_alignment_objectives = latent_alignment_objectives

        # Decoding params
        self.decode_timestamps = decode_timestamps

    # does not overide parent method
    def to_dict(self):
        brain_encoder_config = self.brain_encoder_config.to_dict()
        config = super().to_dict()
        config["brain_encoder_config"] = brain_encoder_config
        return config

    def from_dict(self, config: tp.Dict[str, tp.Any]):
        if "mmd_loss" not in config["latent_alignment_objectives"]:
            config["latent_alignment_objectives"]["mmd_loss"] = 0.0

        if "bins" in config["brain_encoder_config"]:
            self.brain_encoder_config = SpectralConvConfig.from_dict(
                config["brain_encoder_config"]
            )
        else:
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
        self.random_test_size = config["random_test_size"]
        self.seed = config["seed"]
        self.mel_alignment_objectives = config["mel_alignment_objectives"]
        self.latent_alignment_objectives = config["latent_alignment_objectives"]
        self.decode_timestamps = config["decode_timestamps"]
        
        if self.use_adalora:
            self.adalora_config = AdaLoraConfig(
                peft_type="ADALORA",
                task_type="SPEECH_RECOGNITION",
                target_modules=["q_proj", "v_proj"],
                init_r=config["adalora_init_r"],
                target_r=config["adalora_target_r"],
                tinit=config["adalora_tinit"],
                tfinal=config["adalora_tfinal"],
                deltaT=config["adalora_deltaT"],
                lora_alpha=config["adalora_lora_alpha"],
                lora_dropout=config["adalora_lora_dropout"],
                total_step=config["adalora_total_step"],
            )
        else:
            self.adalora_config = None
            
        return self
