from torch import nn
import torch
import math
from .conformer import Conformer
from .transformer import CustomTransformerEncoder, CustomTransformerDecoder


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)  # [L, D]
        positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [L, 1]

        i_vals = torch.arange(0, d_model // 2, dtype=torch.float)  # [D/2]
        freqs = 10000.0 ** (-2 * i_vals / d_model)  # [D/2]

        pe_sin = torch.sin(positions * freqs)  # [L, D/2]
        pe_cos = torch.cos(positions * freqs)  # [L, D/2]

        # Interleave sin and cos
        pe[:, 0::2] = pe_sin  # even
        pe[:, 1::2] = pe_cos  # odd

        self.register_buffer("pe", pe.unsqueeze(0))  # [1, L, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class GroupedConvolution(nn.Module):
    """
    Inspired by wav2vec 2.0, convolution over the input features (z) to
    generate positional embeddings.

    - A convolution with `groups = d_model` is used to produce positional embeddings.
    - After conv, it's added to the input features.
    """

    def __init__(self, d_model: int, kernel_size: int = 15):
        super().__init__()
        # group the conv by d_model channels (depthwise convolution)
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding="same",
            groups=d_model,
        )
        nn.init.kaiming_normal_(self.conv.weight, a=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C] -> [B, C, T]
        x_t = x.transpose(1, 2)
        pos_emb = self.conv(x_t)  # [B, C, T]
        pos_emb = pos_emb.transpose(1, 2)  # [B, T, C]

        return x + pos_emb


class TransformerEmbedding(nn.Module):
    supported_embeddings = ["sinusoidal", "groupconv"]

    def __init__(
        self,
        embedding: str = None,
        d_model: int = 256,
        dropout: float = 0.1,
        max_len: int = 5000,
        kernel_size: int = 4,
    ):
        super().__init__()
        assert (
            embedding in self.supported_embeddings or embedding is None
        ), f"Embedding {embedding} not supported."

        self.embedding_type = embedding

        if embedding == "sinusoidal":
            self.embedding = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        elif embedding == "groupconv":
            self.embedding = GroupedConvolution(d_model, kernel_size=kernel_size)
        else:
            self.embedding = None

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        x = self.embedding(x) if self.embedding is not None else x
        return self.dropout(x)


class RNNEncoder(nn.Module):
    """
    Transformer / Conformer encoder model for time series data with custom embedding options
    Note, no final projection is done here. Optionally, concat a channel-reduced
    spectral embedding to the input features.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float,
        layers: int,
        embedding: str,
        rnn_type: str = "transformer",
        # Conformer params
        depthwise_conv_kernel_size: int = 31,
        use_group_norm: bool = True,
        convolution_first: bool = False,
    ):
        super().__init__()

        assert rnn_type in [
            "transformer",
            "conformer",
        ], f"rnn_type {rnn_type} not supported."

        self.d_model = d_model
        self.embedding_name = embedding
        self.embedding = TransformerEmbedding(
            embedding=embedding, d_model=d_model, dropout=dropout
        )
        self.rnn_type = rnn_type

        if rnn_type == "transformer":
            self.encoders = CustomTransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    batch_first=True,
                    dropout=dropout,
                    dim_feedforward=4 * d_model,
                ),
                num_layers=layers,
            )
        else:
            self.encoders = Conformer(
                input_dim=d_model,
                num_heads=nhead,
                ffn_dim=4 * d_model,
                num_layers=layers,
                depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                use_group_norm=use_group_norm,
                convolution_first=convolution_first,
            )

        total_params = sum(p.numel() for p in self.parameters())
        print(
            f"RNNEncoder initialized as {rnn_type} with {layers} layers, {d_model} d_model, {nhead} nhead"
        )
        print(f"\tEmbedding: {embedding}, params: {total_params}")

    def forward(
        self, x: torch.Tensor, attn_mask=None, return_hidden_outputs: bool = False
    ):
        """
        x: [B, C, T], attn_mask: [B, T]

        returns: [B, C, T], hidden_outputs: [B, C, T] of length layers
        """
        x = self.embedding(x.transpose(1, 2))  # [B, C, T] -> [B, T, C]

        if self.rnn_type == "transformer":
            x, hidden_outputs = self.encoders(
                x,
                mask=None,
                src_key_padding_mask=attn_mask,
                is_causal=False,
                return_hidden_outputs=return_hidden_outputs,
            )  # [B, T, C]

        # Conformer
        else:
            if attn_mask is None:
                lengths = torch.full((x.shape[0],), x.shape[1], dtype=torch.long).to(
                    x.device
                )
            else:
                # Transform mask [B, T] -> [B,]
                lengths = (attn_mask == False).sum(dim=1).to(torch.int32).to(x.device)

            x, _, hidden_outputs = self.encoders(
                x, lengths=lengths, return_hidden_outputs=return_hidden_outputs
            )  # [B, T, C]

        x = x.transpose(1, 2)  # [B, C, T]

        # hidden outputs: [B, T, C] -> [B, C, T]
        if len(hidden_outputs) > 0:
            hidden_outputs = [ho.transpose(1, 2) for ho in hidden_outputs]

        return x, hidden_outputs


class TransformerDecoder(nn.Module):
    """
    Transformer decoder model for Mel prediction
    """

    mel_bins = 80

    def __init__(
        self,
        encoder_output_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        dropout: float = 0.2,
        layers: int = 4,
        embedding: str = None,
    ):
        super().__init__()
        self.encoder_output_dim = encoder_output_dim
        self.encoder_proj = None

        # ENCODER PROJECTION
        if encoder_output_dim != d_model:
            pad, kernel, stride = 0, 1, 1
            self.encoder_proj = nn.Sequential(
                nn.Conv1d(encoder_output_dim, 2 * encoder_output_dim, 1),
                nn.GELU(),
                nn.ConvTranspose1d(
                    2 * encoder_output_dim,
                    d_model,
                    kernel,
                    stride,
                    pad,
                ),
            )

        # DECODER EMBEDDING
        self.embedding_name = embedding
        self.embedding = TransformerEmbedding(
            embedding=embedding, d_model=self.mel_bins, dropout=dropout
        )
        # DECODER PROJECTION, mel_bins to d_model
        self.decoder_proj = None
        if self.mel_bins != d_model:
            self.decoder_proj = nn.Linear(self.mel_bins, d_model)

        self.decoders = CustomTransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                batch_first=True,
                dropout=dropout,
                dim_feedforward=4 * d_model,
            ),
            num_layers=layers,
        )
        total_params = sum(p.numel() for p in self.parameters())
        print(
            f"TransformerDecoder initialized with {layers} layers, {d_model} d_model, {nhead} nhead"
        )
        print(f"\tEmbedding: {embedding}, params: {total_params}")

    def forward(
        self,
        mel: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask=None,
        return_hidden_outputs: bool = False,
    ):
        """Uses the encoder output to predict the mel spectrogram

        Arguments:
            mel -- mel spectrogram of the perceived audio [B, mel, T]
            encoder_output -- output from the encoder [B, C, T]
            src_mask -- mask for the encoder output [B, T]

        Returns:
            [B, d_model, T] -- predicted mel spectrogram
            [B, d_model, T] of len layers -- hidden outputs from the decoder
        """
        # [B, mel, T] -> [B, T, mel]
        mel = self.embedding(mel.transpose(1, 2))
        if self.decoder_proj is not None:
            mel = self.decoder_proj(mel)  # [B, T, d_model]

        if self.encoder_proj is not None:
            encoder_output = self.encoder_proj(encoder_output)  # [B, d_model, T]
        encoder_output = encoder_output.transpose(1, 2)  # [B, T, d_model]

        _, t, _ = mel.shape
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            sz=t
        )  # of shape [T, T]

        output, hidden_outputs = self.decoders(
            tgt=mel,
            memory=encoder_output,
            tgt_mask=causal_mask,  # Causal mask for decoder self-attention
            memory_mask=None,  # No mask for cross-attention with encoder
            tgt_key_padding_mask=None,
            memory_key_padding_mask=src_mask,  # If meg padded
            tgt_is_causal=True,
            memory_is_causal=False,
            return_hidden_outputs=return_hidden_outputs,
        )  # [B, T, d_model]

        output = output.transpose(1, 2)  # [B, d_model, T]

        # hidden outputs: [B, T, d_model] -> [B, d_model, T]
        if len(hidden_outputs) > 0:
            hidden_outputs = [ho.transpose(1, 2) for ho in hidden_outputs]

        return output, hidden_outputs
