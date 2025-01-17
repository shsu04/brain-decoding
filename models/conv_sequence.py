import typing as tp
from torch import nn
import torch


class ConvSequence(nn.Module):
    """
    Convolutional sequence with optional skip connections and GLU activations.
    Raw signal variant, where the input is [B, C, T]
    """

    def __init__(
        self,
        channels: tp.Sequence[int],
        kernel: int = 4,
        dilation_growth: int = 1,
        dilation_period: tp.Optional[int] = None,
        stride: int = 1,
        dropout: float = 0.0,
        decode: bool = False,
        batch_norm: bool = False,
        dropout_input: float = 0,
        glu: int = 0,
        activation: tp.Any = None,
        pos_encoding: bool = False,
        half: bool = False,
    ) -> None:
        """
        Arguments:
            channels -- List of channel dims for each layer.

        Keyword Arguments:
            kernel -- Convolutional kernel size (default: {4})
            dilation_growth -- Growth factor for dilation (default: {1})
            dilation_period -- Period for resetting dilation (default: {None})
            stride -- Convolutional stride (default: {2})
            dropout -- Dropout rate (default: {0.0})
            decode -- If True, uses ConvTranspose1d (default: {False})
            batch_norm -- If True, uses batch normalization (default: {False})
            dropout_input -- Dropout rate for input (default: {0})
            glu -- If > 0, uses GLU activation every `glu` layers (default: {0})
            activation -- Activation function (default: {None})
            pos_encoding -- If True, uses positional encoding over freq and channels (default: {False})
            half -- If True, uses stride 2 for third to last layer (default: {False})
                This downsamples the input by 2x.
        """

        super().__init__()

        dilation = 1
        channels = tuple(channels)
        self.sequence = nn.ModuleList()
        self.glus = nn.ModuleList()

        Conv = nn.Conv1d if not decode else nn.ConvTranspose1d

        self.pos_encoding = False
        if pos_encoding:
            self.pos_encoding = True

            self.chan_embedding = nn.Parameter(
                torch.randn(1, channels[0], 1), requires_grad=True
            )
            nn.kaiming_uniform_(self.chan_embedding, a=0)

        # Build layers
        for k, (chin, chout) in enumerate(zip(channels[:-1], channels[1:])):
            layers: tp.List[nn.Module] = []
            is_last = k == len(channels) - 2

            # Add input dropout if specified for first layer
            if k == 0 and dropout_input:
                assert 0 < dropout_input < 1
                layers.append(nn.Dropout(dropout_input))

            # Add convolutional layer with optional dilation
            if dilation_growth > 1:
                assert kernel % 2 != 0, "Supports only odd kernel with dilation for now"
            if dilation_period and (k % dilation_period) == 0:
                dilation = 1

            kernel_k = (
                (kernel * 2 - 1) if (half and (k < len(channels) - 4)) else kernel
            )
            pad_k = kernel_k // 2 * dilation

            conv_layer = Conv(
                chin,
                chout,
                kernel_k,
                (stride * 2) if (half and (k == len(channels) - 4)) else stride,
                pad_k,
                dilation=dilation,
                groups=1,
            )

            nn.init.kaiming_uniform_(conv_layer.weight, a=0)
            if conv_layer.bias is not None:
                nn.init.zeros_(conv_layer.bias)

            layers.append(conv_layer)

            dilation *= dilation_growth

            # Batch norm, activation, and dropout
            if not is_last:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(num_features=chout))
                layers.append(activation())

                if dropout:
                    layers.append(nn.Dropout(dropout))

            self.sequence.append(nn.Sequential(*layers))

            # Add GLU layer if specified
            if glu and (k + 1) % glu == 0:
                glu_layer = nn.Sequential(
                    nn.Conv1d(
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

    def forward(self, x: tp.Any, return_hidden_outputs: bool = False) -> tp.Any:

        hidden_outputs = []

        if self.pos_encoding:
            x = x + self.chan_embedding

        for module_idx, module in enumerate(self.sequence):

            old_x = x
            x = module(x)

            # Residual
            if x.shape == old_x.shape:
                x = x + old_x

            glu = self.glus[module_idx]

            if glu is not None:
                x = glu(x)

            if return_hidden_outputs:
                hidden_outputs.append(x)

        return x, hidden_outputs
