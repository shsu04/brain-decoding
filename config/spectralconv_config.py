from .config import Config


class SpectralConvConfig(Config):

    def __init__(
        self,
        mel_normalization=False,
        # Str to list of possible conditions
        conditions: dict[str, list] = None,
        # Channels
        in_channels: int = 208,
        out_channels: int = 128,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        initial_batch_norm: bool = False,
        # Sensor layout settings
        layout_dim: int = 2,
        layout_proj: bool = False,
        layout_scaling: str = "midpoint",
        # Merger with spatial attn
        merger: bool = True,
        merger_emb_type: str = "fourier",
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
        cnn_channels: list[int] = [384, 384, 256, 128, 64, 32, 16, 8, 3],
        kernel_size: int = 3,
        dilation_growth: int = 2,
        dilation_period: int = 5,
        glu: int = 1,
        conv_dropout: float = 0.2,
        dropout_input: float = 0.2,
        batch_norm: bool = True,
        half: bool = False,
        cnn_pos_encoding: bool = True,
        bins: int = 128,
        hop_length: int = 2,
        # Quantizer
        quantizer: str = "vq",  # vq or gumbel or none
        num_codebooks: int = 1,
        codebook_size: int = 512,
        quantizer_commitment: float = 0.25,
        quantizer_temp_init: float = 1.0,
        quantizer_temp_min: float = 0.1,
        quantizer_temp_decay: float = 0.999,
        # Transformers Encoders
        transformer_input: str = "concat",  # concat or quantized or continuous
        transformer_encoder_emb: str = "groupconv",
        transformer_encoder_layers: int = 0,
        transformer_encoder_heads: int = 0,
        # Conformer encoder variant
        rnn_type: str = "transformer",
        depthwise_conv_kernel_size: int = 31,
        use_group_norm: bool = True,
        convolution_first: bool = False,
        # Transformer Decoders
        transformer_decoder_emb: str = "groupconv",
        transformer_decoder_layers: int = 0,
        transformer_decoder_heads: int = 0,
        transformer_decoder_dim: int = 0,
    ):
        super(SpectralConvConfig, self).__init__()

        assert dropout < 1.0, "Dropout should be a probability"
        assert merger_dropout < 1.0, "Merger dropout should be a probability"
        assert dropout_input < 1.0, "Dropout input should be a probability"
        assert conv_dropout < 1.0, "Conv dropout should be a probability"

        self.initial_batch_norm = initial_batch_norm
        self.conditions = conditions
        self.mel_normalization = mel_normalization
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
        self.merger_emb_type = merger_emb_type
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
        self.cnn_channels = cnn_channels
        self.kernel_size = kernel_size
        self.dilation_growth = dilation_growth
        self.dilation_period = dilation_period
        self.glu = glu
        self.conv_dropout = conv_dropout
        self.dropout_input = dropout_input
        self.batch_norm = batch_norm
        self.half = half
        self.cnn_pos_encoding = cnn_pos_encoding
        self.bins = bins
        self.hop_length = hop_length
        # Quantizer
        self.quantizer = quantizer
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.quantizer_commitment = quantizer_commitment
        self.quantizer_temp_init = quantizer_temp_init
        self.quantizer_temp_min = quantizer_temp_min
        self.quantizer_temp_decay = quantizer_temp_decay
        # Transformer Encoders
        self.transformer_input = transformer_input
        self.transformer_encoder_emb = transformer_encoder_emb
        self.transformer_encoder_layers = transformer_encoder_layers
        self.transformer_encoder_heads = transformer_encoder_heads
        # Conformer variant
        self.rnn_type = rnn_type
        self.depthwise_conv_kernel_size = depthwise_conv_kernel_size
        self.use_group_norm = use_group_norm
        self.convolution_first = convolution_first
        # Transformer Decoders
        self.transformer_decoder_emb = transformer_decoder_emb
        self.transformer_decoder_layers = transformer_decoder_layers
        self.transformer_decoder_heads = transformer_decoder_heads
        self.transformer_decoder_dim = transformer_decoder_dim
