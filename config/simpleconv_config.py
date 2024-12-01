import typing as tp
from .config import Config


class SimpleConvConfig(Config):
    def __init__(
        self,
        use_attention_mask: bool = False,
        is_causal: bool = False,
        transformer_layers: int = 0,
        transformer_heads: int = 8,
        # Channels
        channels: int = 208,
        out_channels: int = 128,
        hidden: int = 512,  # 320
        # Overall structure
        depth: int = 5,
        concatenate: bool = False,  # concatenate the inputs
        linear_out: bool = False,
        complex_out: bool = True,
        # Conv layer
        kernel_size: int = 3,
        growth: float = 1.0,
        dilation_growth: int = 2,
        dilation_period: tp.Optional[int] = 5,
        skip: bool = True,
        post_skip: bool = False,
        scale: tp.Optional[float] = None,
        rewrite: bool = False,
        groups: int = 1,
        glu: int = 1,
        glu_context: int = 1,
        glu_glu: bool = True,
        gelu: bool = True,
        # Dual path RNN
        dual_path: int = 0,
        # Dropouts, BN, activations
        conv_dropout: float = 0.3,
        dropout_input: float = 0.3,
        batch_norm: bool = True,
        relu_leakiness: float = 0.0,
        # Subject specific settings
        n_subjects: int = 27,
        subject_dim: int = 0,
        subject_layers: bool = True,
        subject_layers_dim: str = "input",  # or hidden
        subject_layers_id: bool = False,
        embedding_scale: float = 1.0,
        # stft transform
        n_fft: tp.Optional[int] = None,
        fft_complex: bool = True,
        # Attention multi-dataset support
        merger: bool = True,
        merger_pos_dim: int = 2048,
        merger_channels: int = 256,
        merger_dropout: float = 0.3,
        merger_penalty: float = 0.0,
        merger_per_subject: bool = False,
        dropout: float = 0.3,
        dropout_rescale: bool = True,
        initial_linear: int = 256,
        initial_depth: int = 1,
        initial_nonlin: bool = False,
        subsample_meg_channels: int = 0,
    ):
        super(SimpleConvConfig, self).__init__()

        self.use_attention_mask = use_attention_mask
        self.is_causal = is_causal
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        # Channels
        self.in_channels = {"meg": channels}
        self.channels = channels
        self.out_channels = out_channels
        self.hidden = {"meg": hidden}
        # Overall structure
        self.depth = depth
        self.concatenate = concatenate
        self.linear_out = linear_out
        self.complex_out = complex_out
        # Conv layer
        self.kernel_size = kernel_size
        self.growth = growth
        self.dilation_growth = dilation_growth
        self.dilation_period = dilation_period
        self.skip = skip
        self.post_skip = post_skip
        self.scale = scale
        self.rewrite = rewrite
        self.groups = groups
        self.glu = glu
        self.glu_context = glu_context
        self.glu_glu = glu_glu
        self.gelu = gelu
        # Dual path RNN
        self.dual_path = dual_path
        # Dropouts, BN, activations
        self.conv_dropout = conv_dropout
        self.dropout_input = dropout_input
        self.batch_norm = batch_norm
        self.relu_leakiness = relu_leakiness
        # Subject specific settings
        self.n_subjects = n_subjects
        self.subject_dim = subject_dim
        self.subject_layers = subject_layers
        self.subject_layers_dim = subject_layers_dim
        self.subject_layers_id = subject_layers_id
        self.embedding_scale = embedding_scale
        # stft transform
        self.n_fft = n_fft
        self.fft_complex = fft_complex
        # Attention multi-dataset support
        self.merger = merger
        self.merger_pos_dim = merger_pos_dim
        self.merger_channels = merger_channels
        self.merger_dropout = merger_dropout
        self.merger_penalty = merger_penalty
        self.merger_per_subject = merger_per_subject
        self.dropout = dropout
        self.dropout_rescale = dropout_rescale
        self.initial_linear = initial_linear
        self.initial_depth = initial_depth
        self.initial_nonlin = initial_nonlin
        self.subsample_meg_channels = subsample_meg_channels
