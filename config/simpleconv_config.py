import typing as tp
from .config import Config


class SimpleConvConfig(Config):
    def __init__(
        self,
        # Str to list of possible conditions
        conditions: dict[str, list] = None,
        # Channels
        in_channels: int = 208,
        out_channels: int = 128,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        # Sensor layout settings
        layout_dim: int = 2,
        layout_proj: bool = False,
        layout_scaling: str = "midpoint",
        # Merger with spatial attn
        merger: bool = True,
        merger_emb_dim: int = 2048,
        merger_channels: int = 256,
        merger_dropout: float = 0.3,
        merger_conditional: str = None,
        # Inital
        initial_linear: int = 256,
        initial_depth: int = 1,
        # Conditional layers
        conditional_layers: bool = True,
        conditional_layers_dim: str = "input",  # or hidden_dim
        # Conv layer overall structure
        depth: int = 5,
        kernel_size: int = 3,
        growth: float = 1.0,
        dilation_growth: int = 2,
        dilation_period: tp.Optional[int] = 5,
        glu: int = 1,
        conv_dropout: float = 0.2,
        dropout_input: float = 0.2,
        batch_norm: bool = True,
        # Transformers Encoders
        transformer_encoder_emb: str = "grouped",
        transformer_encoder_layers: int = 0,
        transformer_encoder_heads: int = 0,
        transformer_encoder_concat_spectrals: bool = False,
        transformer_encoder_bins: int = None,
        # Transformer Decoders
        transformer_decoder_emb: str = "grouped",
        transformer_decoder_layers: int = 0,
        transformer_decoder_heads: int = 0,
        transformer_decoder_dim: int = 0,
    ):
        super(SimpleConvConfig, self).__init__()

        self.conditions = conditions
        # Channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        # Sensor layout settings
        self.layout_dim = layout_dim
        self.layout_proj = layout_proj
        self.layout_scaling = layout_scaling
        # Merger with spatial attn
        self.merger = merger
        self.merger_emb_dim = merger_emb_dim
        self.merger_channels = merger_channels
        self.merger_dropout = merger_dropout
        self.merger_conditional = merger_conditional
        # Inital
        self.initial_linear = initial_linear
        self.initial_depth = initial_depth
        # Subject specific settings
        self.conditional_layers = conditional_layers
        self.conditional_layers_dim = conditional_layers_dim
        # Conv layer overall structure
        self.depth = depth
        self.kernel_size = kernel_size
        self.growth = growth
        self.dilation_growth = dilation_growth
        self.dilation_period = dilation_period
        self.glu = glu
        self.dropout_input = dropout_input
        self.conv_dropout = conv_dropout
        self.batch_norm = batch_norm
        # Transformer Encoders
        self.transformer_encoder_emb = transformer_encoder_emb
        self.transformer_encoder_layers = transformer_encoder_layers
        self.transformer_encoder_heads = transformer_encoder_heads
        self.transformer_encoder_concat_spectrals = transformer_encoder_concat_spectrals
        self.transformer_encoder_bins = transformer_encoder_bins
        # Transformer Decoders
        self.transformer_decoder_emb = transformer_decoder_emb
        self.transformer_decoder_layers = transformer_decoder_layers
        self.transformer_decoder_heads = transformer_decoder_heads
        self.transformer_decoder_dim = transformer_decoder_dim
