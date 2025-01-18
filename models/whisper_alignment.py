import torch
import torch.nn as nn
from transformers import WhisperModel
from typing import List, Optional
from config import SimpleConvConfig, SpectralConvConfig
from models.simpleconv import SimpleConv
from models.spectralconv import SpectralConv
import typing as tp
from studies.study import Recording
from peft import AdaLoraConfig, AdaLoraModel


class WhisperAlignment(nn.Module):
    def __init__(
        self,
        brain_module_config: tp.Union[SimpleConvConfig, SpectralConvConfig],
        adalora_config: AdaLoraConfig,
        layers_to_align: Optional[List[int]] = [-1],
        use_compile: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Uses the encoder hidden states from a pre-trained model for alignment

        Arguments:
            simpleconv_config -- Brain encoder config

        Keyword Arguments:
            layers_to_align -- which hidden layers to output for alignment
            use_compile -- compile can have 4.5x speedup
        """
        super().__init__()

        if isinstance(brain_module_config, SimpleConvConfig):
            self.brain_module = SimpleConv(brain_module_config)
        elif isinstance(brain_module_config, SpectralConvConfig):
            self.brain_module = SpectralConv(brain_module_config)

        self.model_id = "openai/whisper-large-v3"

        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        whisper_model = WhisperModel.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(device)

        # Only encoder is used for alignment, free mem
        self.encoder = whisper_model.get_encoder()
        del whisper_model.decoder
        del whisper_model

        self.freeze()

        # Which hidden layers to align, last by default
        self.layers_to_align = layers_to_align
        assert all([i < 32 for i in layers_to_align]), "Invalid layer index"

        if use_compile:
            self.compile()

        self.device = device
        self.to(device)
        self.half() if torch.cuda.is_available() else self.float()

        # AdaLora
        self.adalora_config = adalora_config
        self.encoder = AdaLoraModel(
            self.encoder, adalora_config, adapter_name="default"
        )

    def compile(self):
        """Only used when inference is done"""
        self.encoder.forward = torch.compile(
            self.encoder.forward, mode="reduce-overhead", fullgraph=True
        )

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(
        self,
        x: tp.List[torch.Tensor],
        recording: tp.List[Recording],
        conditions: tp.List[tp.Dict[str, str]] = None,
        mel: tp.List[torch.Tensor] = None,
        train: bool = False,
        return_hidden_outputs: bool = False,
    ) -> tuple[List[torch.Tensor], torch.Tensor]:
        """
        x: [B, C, T]

        Returns:
            x - predicted mel [B, 128, 3000]
            Quantizer metrics [B, 80, 3000]
            Channel weights [B, C, C']
            Hidden outputs [B, 80, 3000] of length brain encoder layers

            List of hidden states for each encoder layer in layers_to_align [B, T, D]
            Last hidden state [B, T, D]
            Where 1500 = T, 1280 = D
        """

        x, quantizer_metrics, channel_weights, hidden_outputs = self.brain_module(
            x, recording, conditions, mel, train, return_hidden_outputs
        )  # [B, 80, T]
        B, C, T = x.size()

        assert C == 128, f"Expected {128} channels, got {C}"
        assert T == 3000, f"Expected {3000} timesteps, got {T}"

        # Pad or truncate
        if T < 3000:
            pad_len = 3000 - T
            x = nn.functional.pad(x, (0, pad_len), mode="constant", value=0.0)
        elif T > 3000:
            # If longer, trim
            x = x[:, :, :3000]

        encoder_outputs = self.encoder(x, output_hidden_states=True)

        return (
            x,
            quantizer_metrics,
            channel_weights,
            hidden_outputs,
            [encoder_outputs.hidden_states[i] for i in self.layers_to_align],
            encoder_outputs.last_hidden_state,
        )
