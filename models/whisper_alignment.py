import gc
import torch
import torch.nn as nn
from transformers import WhisperModel
from typing import List, Optional
from config import SimpleConvConfig, SpectralConvConfig
from models.simpleconv import SimpleConv
from models.spectralconv import SpectralConv
import typing as tp
from studies.study import Recording
from peft import AdaLoraConfig, get_peft_model


class WhisperAlignment(nn.Module):
    def __init__(
        self,
        brain_module_config: tp.Union[SimpleConvConfig, SpectralConvConfig],
        adalora_config: AdaLoraConfig,
        layers_to_align: Optional[List[int]] = [-1],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Uses the encoder hidden states from a pre-trained model for alignment

        Arguments:
            simpleconv_config -- Brain encoder config

        Keyword Arguments:
            layers_to_align -- which hidden layers to output for alignment
        """
        super().__init__()

        if isinstance(brain_module_config, SimpleConvConfig):
            self.brain_module_config = brain_module_config
            self.brain_module = SimpleConv(brain_module_config)
        elif isinstance(brain_module_config, SpectralConvConfig):
            self.brain_module_config = brain_module_config
            self.brain_module = SpectralConv(brain_module_config)

        self.model_id = "openai/whisper-base"

        if torch.cuda.get_device_capability() in [(7, 0), (8, 0), (9, 0)]:
            torch_dtype = torch.bfloat16
        elif torch.cuda.is_available():
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        whisper_model = WhisperModel.from_pretrained(
            self.model_id,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            torch_dtype=torch_dtype,
        ).to(device)

        # Only encoder is used for alignment, free mem
        encoder = whisper_model.get_encoder()
        encoder._freeze_parameters()
        del whisper_model.decoder
        del whisper_model

        torch.cuda.empty_cache()
        gc.collect()

        # Which hidden layers to align, last by default
        self.layers_to_align = layers_to_align
        assert all([i < 32 for i in layers_to_align]), "Invalid layer index"

        # AdaLora
        self.adalora_config = adalora_config
        prefixes = ["layers"]
        suffixes = ["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"]

        target_modules = self.match_modules_string(
            encoder.named_modules(), prefixes, suffixes
        )
        print(f"Found {len(target_modules)} target modules for AdaLora: {suffixes}")
        self.adalora_config.target_modules = target_modules
        self.encoder = get_peft_model(encoder, adalora_config)

        print(
            f"AdaLora model has {sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)} parameters"
        )

        self.device = device
        self.to(device)

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
        Arguments:
            x -- meg scans of shape [B, C, T]
            recording -- Recording object with the layout and subject index
            conditions -- dictionary of conditions_type : condition_name
            mel -- mel spectrogram of shape [B, mel_bins, T], UNSHIFTED.
            train -- boolean flag to indicate training or inference
            return_hidden_outputs -- flag to return hidden outputs from CNN and RNNs, [B, C, T] of length L

        Returns:
            x - predicted mel [B, 80, 3000]
            Quantizer metrics [B, 80, 3000]
            Channel weights [B, C, C']
            Hidden outputs [B, 80, 3000] of length brain encoder layers

            List of hidden states for each encoder layer in layers_to_align [B, T, D]
            Where 1500 = T, 1280 = D
        """

        x, quantizer_metrics, channel_weights, hidden_outputs = self.brain_module(
            x, recording, conditions, mel, train, return_hidden_outputs
        )  # [B, 80, T]
        B, C, T = x.size()

        assert C == 80, f"Expected {80} channels, got {C}"

        # Pad or truncate
        if T < 3000:
            pad_len = 3000 - T
            x = nn.functional.pad(x, (0, pad_len), mode="constant", value=0.0)
        elif T > 3000:
            # If longer, trim
            x = x[:, :, :3000]

        # Set up hidden states
        if len(self.layers_to_align) is None:
            return x, quantizer_metrics, channel_weights, None, None
        elif self.layers_to_align == [-1]:
            output_hidden_states = False
        else:
            output_hidden_states = True

        # Whisper
        encoder_outputs = self.encoder(x, output_hidden_states=output_hidden_states)

        x = x[:, :, :T]  # Trim padding

        # Sort hidden states, trim to time step
        if self.layers_to_align == [-1] or self.config.latent_alignment_layers == [32]:
            hidden_states = [encoder_outputs.last_hidden_state[:, :T, :]]
        else:
            hidden_states = [
                encoder_outputs.hidden_states[i][:, :T, :] for i in self.layers_to_align
            ]
            del encoder_outputs.hidden_states

        gc.collect()
        torch.cuda.empty_cache()

        return (
            x,
            quantizer_metrics,
            channel_weights,
            hidden_outputs,
            hidden_states,
        )

    def match_modules_string(
        self,
        named_modules,
        start_prefixes,
        end_suffixes,
        mid_prefixes=[],
    ):
        matched_modules = []

        for name, _ in named_modules:

            start_matched = False
            for start in start_prefixes:
                if name.startswith(start):
                    start_matched = True
                    break

            if not start_matched:
                continue

            if mid_prefixes:
                mid_matched = False
                for mid in mid_prefixes:
                    if mid in name:
                        mid_matched = True
                        break

                if not mid_matched:
                    continue

            end_matched = False
            for end in end_suffixes:
                if name.endswith(end):
                    matched_modules.append(name)
                    end_matched = True
                    break

            if not end_matched:
                continue

        return matched_modules
