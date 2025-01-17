from turtle import pos
import torch
import torch.nn as nn
import typing as tp


class SpectralConvSequence(nn.Module):

    def __init__(
        self,
        channels: tp.Sequence[int],
        kernel: int = 4,
        dilation_growth: int = 1,
        dilation_period: tp.Optional[int] = None,
        stride: int = 1,
        dropout: float = 0.0,
        decode: bool = False,
        group_norm: bool = True,
        dropout_input: float = 0,
        glu: int = 0,
        activation: tp.Any = None,
        half: bool = False,
        pos_encoding: bool = False,
        mels: int = 0,
    ):
        """
        Spectrogram conv variant using positional encoding over the freq and channels
        same as ConvSequence, but has input [B, C, mel, T] instead of [B, C, T], using
        2D variants.

        Arguments:
            channels -- List of channel dims for each layer.

        Keyword Arguments:
            kernel -- Convolutional kernel size (default: {4})
            dilation_growth -- Growth factor for dilation (default: {1})
            dilation_period -- Period for resetting dilation (default: {None})
            stride -- Convolutional stride (default: {2})
            dropout -- Dropout rate (default: {0.0})
            decode -- If True, uses ConvTranspose1d (default: {False})
            group_norm -- If True, uses group normalization (default: {False})
            dropout_input -- Dropout rate for input (default: {0})
            glu -- If > 0, uses GLU activation every `glu` layers (default: {0})
            activation -- Activation function (default: {None})
            half -- If True, uses stride 2 for third to last layer (default: {False})
                This downsamples the input by 2x.
            pos_encoding -- If True, uses positional encoding over freq and channels (default: {False})
            mels -- Number of mel bins for positional encoding (default: {0})

        Outputs [B, C, mel, T] with smaller C for flattening to [B, C * mel, T]
        """
        super().__init__()
        dilation = 1
        channels = tuple(channels)
        self.sequence = nn.ModuleList()
        self.skip_conv = nn.ModuleList()
        self.glus = nn.ModuleList()

        Conv = nn.Conv2d if not decode else nn.ConvTranspose2d

        # Different to SimpleConv, we have positional encoding over the freq and channels
        self.pos_encoding = False
        if pos_encoding:
            assert mels > 0, "Pos encoding mels must be greater than 0"

            self.pos_encoding = True

            self.chan_embedding = nn.Parameter(
                torch.zeros(1, channels[0], 1, 1), requires_grad=True
            )
            self.freq_embedding = nn.Parameter(
                torch.zeros(1, 1, mels, 1), requires_grad=True
            )

            nn.init.kaiming_uniform_(self.chan_embedding, a=0)
            nn.init.kaiming_uniform_(self.freq_embedding, a=0)

        # Build layers
        for k, (chin, chout) in enumerate(zip(channels[:-1], channels[1:])):
            layers: tp.List[nn.Module] = []
            is_last = k == len(channels) - 2

            # Add input dropout if specified for first layer
            if k == 0 and dropout_input:
                assert 0 < dropout_input < 1
                layers.append(nn.Dropout(dropout_input))

            # Add dialation (across time) if specified
            if dilation_growth > 1:
                assert kernel % 2 != 0, "Supports only odd kernel with dilation for now"
            if dilation_period and (k % dilation_period) == 0:
                dilation = 1

            # Kernel, pad, and stride across time, halve using double stride size
            # In this case, kernel is also doubled to have appropriate receptive field
            time_kernel = (
                (kernel * 2 - 1) if (half and (k < len(channels) - 4)) else kernel
            )
            time_pad = time_kernel // 2 * dilation
            time_stride = (
                (stride * 2) if (half and (k == len(channels) - 4)) else stride
            )

            # In/out dims for freq remains identical
            freq_kernel, freq_pad, freq_stride = (
                kernel,
                kernel // 2 * dilation,
                stride,
            )

            # Create conv layer
            conv_layer = Conv(
                chin,
                chout,
                (freq_kernel, time_kernel),
                (freq_stride, time_stride),
                (freq_pad, time_pad),
                dilation=dilation,
                groups=1,
            )
            nn.init.kaiming_uniform_(conv_layer.weight, a=0)
            layers.append(conv_layer)
            dilation *= dilation_growth

            # group norm, activation, and dropout
            if not is_last:
                if group_norm:
                    layers.append(nn.GroupNorm(num_groups=4, num_channels=chout))
                layers.append(activation())

                if dropout:
                    layers.append(nn.Dropout(dropout))

            self.sequence.append(nn.Sequential(*layers))

            # Skip connection adaptor if in/out channel or count stride across time different
            if chin != chout or time_stride != 1:
                self.skip_conv.append(
                    Conv(chin, chout, 1, (freq_stride, time_stride), 0, groups=1)
                )
            else:
                self.skip_conv.append(None)

            # Add GLU layer if specified (conv2d variant)
            if glu and (k + 1) % glu == 0:
                glu_layer = nn.Sequential(
                    nn.Conv2d(
                        in_channels=chout,
                        out_channels=chout * 2,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.GLU(dim=1),
                )
                nn.init.kaiming_uniform_(glu_layer[0].weight, a=0)
                self.glus.append(glu_layer)
            else:
                self.glus.append(None)

    def forward(
        self, x: torch.Tensor, return_hidden_outputs: bool = False  # [B, C, mel, T]
    ):
        hidden_outputs = []

        if self.pos_encoding:
            x = x + self.chan_embedding + self.freq_embedding  # [B, C, mel, T]

        for module_idx, module in enumerate(self.sequence):

            old_x = x
            x = module(x)  # [B, C_i, mel, T]

            # Residual
            if self.skip_conv[module_idx] is not None:
                x = x + self.skip_conv[module_idx](old_x)
            else:
                x = x + old_x

            # GLU
            glu = self.glus[module_idx]
            if glu is not None:
                x = glu(x)

            if return_hidden_outputs:
                hidden_outputs.append(x)

        return x, hidden_outputs
