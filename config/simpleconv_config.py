import typing as tp
from .config import Config


class SimpleConvConfig(Config):
    def __init__(
        self,
        # Channels
        in_channels: int = 208,
        out_channels: int = 128,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        # Merger with spatial attn
        merger: bool = True,
        merger_emb_dim: int = 2048,
        merger_channels: int = 256,
        merger_dropout: float = 0.3,
        merger_conditional: bool = False,
        merger_conditions: int = 0,
        # Inital
        initial_linear: int = 256,
        initial_depth: int = 1,
        # Subject specific settings
        n_conditions: int = 27,
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
        # Transformers
        use_attention_mask: bool = False,
        is_causal: bool = False,
        transformer_layers: int = 0,
        transformer_heads: int = 0,
    ):
        super(SimpleConvConfig, self).__init__()

        # Channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        # Merger with spatial attn
        self.merger = merger
        self.merger_emb_dim = merger_emb_dim
        self.merger_channels = merger_channels
        self.merger_dropout = merger_dropout
        self.merger_conditional = merger_conditional
        self.merger_conditions = merger_conditions
        # Inital
        self.initial_linear = initial_linear
        self.initial_depth = initial_depth
        # Subject specific settings
        self.n_conditions = n_conditions
        self.conditional_layers = conditional_layers
        self.conditional_layers_dim = conditional_layers_dim
        # Conv layer overall structure
        self.depth = depth
        self.kernel_size = kernel_size
        self.growth = growth
        self.dilation_growth = dilation_growth
        self.dilation_period = dilation_period
        self.glu = glu
        self.conv_dropout = conv_dropout
        self.dropout_input = dropout_input
        self.batch_norm = batch_norm
        # Transformers
        self.use_attention_mask = use_attention_mask
        self.is_causal = is_causal
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
