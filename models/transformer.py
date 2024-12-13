from torch import nn
import torch
import math
import torchaudio.transforms as T


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)  # [L, D]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [L, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # [D/2]

        pe[:, 0::2] = torch.sin(position * div_term)  # [L, D/2]
        pe[:, 1::2] = torch.cos(position * div_term)  # [L, D/2]

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

    def __init__(self, d_model: int, kernel_size: int = 4):
        super().__init__()
        # group the conv by d_model channels (depthwise convolution)
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C] -> [B, C, T]
        x_t = x.permute(0, 2, 1)
        pos_emb = self.conv(x_t)  # [B, C, T]
        pos_emb = pos_emb.permute(0, 2, 1)  # [B, T, C]
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


class SpectralEmbedding(nn.Module):
    """
    Merges a time series [B, C, T] channel-wise into [B, c, T] using attention,
    then compute spectrals for each channel [B, c, T] -> [B, c * bins, T]
    """

    def __init__(
        self,
        bins: int = 16,
        channels: int = 256,
    ):
        super().__init__()
        assert channels % bins == 0, "Channels must be divisible by bins."
        assert bins > 1, "Bins must be greater than 1."

        self.bins = bins
        self.channels = channels

        self.key = nn.Linear(channels, int(channels / bins), bias=False)
        self.query = nn.Linear(channels, channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)

        n_fft = 2 * (bins - 1)
        self.spectrogram_transform = T.Spectrogram(
            n_fft=n_fft,
            hop_length=1,
            normalized=True,
            power=2,
        )

    def forward(self, x: torch.Tensor):
        B, C, T = x.shape
        x = x.permute(0, 2, 1)  # [B, C, T] -> [B, T, C]

        key = self.key(x).transpose(1, 2)  # [B, T, c] -> [B, c, T]
        query = self.query(x).transpose(1, 2)  # [B, T, C] -> [B, C, T]
        value = self.value(x).transpose(1, 2)  # [B, T, C] -> [B, C, T]

        # [B, c, T] x [B, C, T] → [B, c, C]
        scores = torch.einsum("bct,bCt->bcC", key, query)
        scores = torch.softmax(scores / (C**0.5), dim=-1)

        # [B, c, C] x [B, C, T] → [B, c, T]
        out = torch.einsum("bCt,bcC->bct", value, scores)
        # [B, c, T] -> [B, c, bins, T + 1]
        spectral_out = self.spectrogram_transform(out)
        # remove the last time step to match the input shape
        spectral_out = spectral_out[:, :, :, :-1]

        B, c, bins, T = spectral_out.shape
        spectral_out = spectral_out.reshape(B, c * bins, T)
        # [B, c * bins, T]
        return spectral_out


class TransformerEncoder(nn.Module):
    """
    Transformer encoder model for time series data with custom embedding options
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
        concat_spectrals: bool = False,
        bins: int = 16,
        spectral_dim=128,
    ):
        super().__init__()
        self.spectral = None
        if concat_spectrals:
            self.spectral = SpectralEmbedding(bins=bins, channels=spectral_dim)
            d_model += spectral_dim

        self.embedding_name = embedding
        self.embedding = TransformerEmbedding(
            embedding=embedding, d_model=d_model, dropout=dropout
        )
        self.encoders = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                batch_first=True,
                dropout=dropout,
                dim_feedforward=4 * d_model,
            ),
            num_layers=layers,
        )
        self.d_model = d_model

        total_params = sum(p.numel() for p in self.parameters())
        print(f"\nTransEncoder \tParams: {total_params}")
        print(
            f"\t\tSpec: {spectral_dim}, \t\tEmb: {self.embedding_name}, \tBins: {bins}"
        )
        print(f"\t\tLayers: {layers}, \t\tD_model: {d_model}, \t\tNhead: {nhead}")

    def forward(self, x: torch.Tensor, attn_mask=None):
        """x: [B, C, T], attn_mask: [B, T]"""
        if self.spectral is not None:
            spectral = self.spectral(x)  # [B, c * bins, T]
            assert (
                spectral.shape == x.shape
            ), f"Spectral shape mismatch. {spectral.shape} != {x.shape}"
            x = torch.cat([x, spectral], dim=1)  # [B, 2 * d_model, T]

        x = self.embedding(x.permute(0, 2, 1))  # [B, C, T] -> [B, T, C]

        x = self.encoders(
            x,
            mask=None,
            src_key_padding_mask=attn_mask,
            is_causal=False,
        )  # [B, T, C]
        x = x.permute(0, 2, 1)  # [B, C, T]

        return x


class TransformerDecoder(nn.Module):
    """
    Transformer decoder model for Mel prediction
    """

    mel_bins = 128

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

        self.decoders = nn.TransformerDecoder(
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
        print(f"\nTransDecoder \tParams: {total_params}")
        print(
            f"\t\tEmb: {self.embedding_name}, \tLayers: {layers}, \t\tD_model: {d_model}"
        )
        print(f"\t\tNhead: {nhead}, \t\tMel_bins: {self.mel_bins}")

    def forward(self, mel: torch.Tensor, encoder_output: torch.Tensor, src_mask=None):
        """Uses the encoder output to predict the mel spectrogram

        Arguments:
            mel -- mel spectrogram of the perceived audio [B, mel, T]
            encoder_output -- output from the encoder [B, C, T]
            src_mask -- mask for the encoder output [B, T]
        """
        # [B, mel, T] -> [B, T, mel]
        mel = self.embedding(mel.permute(0, 2, 1))
        if self.decoder_proj is not None:
            mel = self.decoder_proj(mel)  # [B, T, d_model]

        if self.encoder_proj is not None:
            encoder_output = self.encoder_proj(encoder_output)  # [B, d_model, T]
        encoder_output = encoder_output.permute(0, 2, 1)  # [B, T, d_model]

        _, t, _ = mel.shape
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            sz=t
        )  # of shape [T, T]

        output = self.decoders(
            mel,
            encoder_output,
            tgt_mask=causal_mask,  # Causal mask for decoder self-attention
            memory_mask=None,  # No mask for cross-attention with encoder
            tgt_key_padding_mask=None,
            memory_key_padding_mask=src_mask,  # If meg padded
            is_causal=True,
        )  # [B, T, d_model]

        output = output.permute(0, 2, 1)  # [B, d_model, T]

        return output
