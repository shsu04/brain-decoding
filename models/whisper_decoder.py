import gc
import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration
from typing import List, Optional
from config import SimpleConvConfig, SpectralConvConfig
from models.simpleconv import SimpleConv
from models.spectralconv import SpectralConv
import typing as tp
from studies.study import Recording
from peft import AdaLoraConfig, get_peft_model
from losses import CLIPLoss


class WhisperDecoder(nn.Module):
    """
    Full encoder-decoder Whisper model for cross-entropy on text tokens,
    plus alignment of MEG-derived mel + final encoder hidden state.

    'labels' can be passed for token-level cross-entropy calculation.
    """

    def __init__(
        self,
        brain_module_config: tp.Union[SimpleConvConfig, SpectralConvConfig],
        adalora_config: AdaLoraConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        audio_model_id: str = "openai/whisper-tiny.en",
    ):
        super().__init__()

        # Brain module
        if isinstance(brain_module_config, SimpleConvConfig):
            self.brain_module_config = brain_module_config
            self.brain_module = SimpleConv(brain_module_config)

        elif isinstance(brain_module_config, SpectralConvConfig):
            self.brain_module_config = brain_module_config
            self.brain_module = SpectralConv(brain_module_config)

        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            # Ampere or beyond uses bfloat16
            if major >= 8:
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        self.audio_model_id = audio_model_id

        # Whisper model, with decoder
        whisper_model = WhisperForConditionalGeneration.from_pretrained(
            self.audio_model_id,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            torch_dtype=torch_dtype,
        ).to(device)

        # AdaLora target modules
        prefixes = ["model.decoder.layers", "model.decoder.layers"]
        suffixes = ["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"]

        target_modules = self.match_modules_string(
            whisper_model.named_modules(), prefixes, suffixes
        )
        print(
            f"Found {len(target_modules)} target modules for AdaLora: {target_modules}"
        )
        self.adalora_config = adalora_config
        self.adalora_config.target_modules = target_modules
        self.decoder = get_peft_model(whisper_model, self.adalora_config)

        # Freeze everything except LoRA
        self.decoder.requires_grad_(False)
        for name, param in self.decoder.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True

        self.d_model = self.decoder.base_model.config.d_model
        self.clip_loss = CLIPLoss(dim=self.decoder.config.d_model)

        self.device = device
        self.to(device)

        print(
            f"{self.audio_model_id} loaded with total params = "
            f"{sum(p.numel() for p in self.decoder.parameters())}."
        )
        trainable = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        print(f"AdaLora has {trainable} trainable params after target module setup.")

        # # Simple Linear layer for studies, [B, C, T] -> [B, C', T]
        # self.gwilliams2023_linear = nn.Conv1d(
        #     208, 256, kernel_size=1, stride=1, padding=0, bias=False
        # )
        # self.armeini2022_linear = nn.Conv1d(
        #     269, 256, kernel_size=1, stride=1, padding=0, bias=False
        # )

    def pad_channels(
        self, x: list[torch.Tensor], desired_channels: int, pad_value: float = 0.0
    ) -> List[torch.Tensor]:
        out = []

        for x_i in x:
            B, C, T = x_i.shape
            if C < desired_channels:
                pad_channels = desired_channels - C
                pad_tensor = torch.full(
                    (B, pad_channels, T), pad_value, device=x_i.device, dtype=x_i.dtype
                )
                x_i = torch.cat([x_i, pad_tensor], dim=1)
            elif C > desired_channels:
                x_i = x_i[:, :desired_channels, :]
            out.append(x_i)

        return out

    def linear_layer(
        self, x: List[torch.Tensor], recording: List[Recording]
    ) -> List[torch.Tensor]:
        out = []
        for i in range(len(x)):
            if recording[i].study_name == "gwilliams2023":
                out.append(self.gwilliams2023_linear(x[i]))
            elif recording[i].study_name == "armeini2022":
                out.append(self.armeini2022_linear(x[i]))
            else:
                raise ValueError(f"Unknown study name: {recording[0].study_name}")

    def forward(
        self,
        x: tp.List[torch.Tensor],
        recording: tp.List[Recording],
        conditions: tp.List[tp.Dict[str, str]] = None,
        mel: tp.List[torch.Tensor] = None,
        train: bool = False,
        return_hidden_outputs: bool = False,
        attention_mask=None,  # For the *input_features* (predicted mel)
        labels: Optional[torch.Tensor] = None,  # For CE loss
        decoder_attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for alignment + token-level cross-entropy.

        Args:
            x: MEG data, list of shape [B, C, T]
            recording: list of Recording object with the layout and subject index
            conditions: optional condition dict of conditions_type : condition_name
            mel: ground-truth mel for alignment, shape [B, 80, T_mel]
            train: boolean flag to indicate training or inference
            return_hidden_outputs: if True, returns the brain_module intermediate states
            attention_mask: optional attention mask for the *encoder input* (mel)
            labels: token IDs for cross-entropy. If provided, model returns .loss
            decoder_attention_mask: optional attention mask for decoder tokens
        Returns:
            (
              predicted_mel: [B, 80, 3000] (brain_module output),
              quantizer_metrics: if quantizer used
              channel_weights: optional from conv, [B, C, C']
              hidden_outputs: optional list of conv layers
              encoder_last_hidden_state: final encoder hidden state from the whisper model, [B, T, d_model]
              ce_loss: cross-entropy if labels' is provided, else None
            )
        """
        # # Option 1: Pad to the same number of channels [B, C, T] -> [B, C', T]
        # x = self.pad_channels(x, desired_channels=269, pad_value=0.0)

        # # Option 2: Simple Linear layer [B, C, T] -> [B, C', T]

        x, quantizer_metrics, channel_weights, hidden_outputs = self.brain_module(
            x, recording, conditions, mel, train, return_hidden_outputs
        )  # [B, 80, T']

        B, C, T = x.size()

        # Pad or truncate to e.g. 3000 mel frames
        x = self.pad_truncate(x, max_frames=3000)

        outputs = self.decoder(
            input_features=x,  # shape [B, 80, max_frames]
            attention_mask=attention_mask,  # optional
            labels=labels,  # yields .loss if not None
            decoder_attention_mask=decoder_attention_mask,
            output_hidden_states=False,  # we only want final encoder hidden state
            return_dict=True,
        )

        # final encoder hidden state -> outputs.encoder_last_hidden_state
        encoder_last_hidden_state = outputs.encoder_last_hidden_state

        ce_loss = outputs.loss if labels is not None else None

        gc.collect()
        torch.cuda.empty_cache()

        return (
            x,  # predicted mel
            quantizer_metrics,
            channel_weights,
            hidden_outputs,
            encoder_last_hidden_state,
            ce_loss,
        )

    @torch.no_grad()
    def generate(
        self,
        x: Optional[torch.Tensor] = None,
        recording: tp.List[Recording] = None,
        conditions: tp.List[tp.Dict[str, str]] = None,
        mel: tp.List[torch.Tensor] = None,
        max_new_tokens: int = 128,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_outputs: bool = False,
        **gen_kwargs,
    ):
        """
        If `mel` is provided, we skip the brain_module and directly use mel
        as the encoder input. Otherwise, we interpret `x` as MEG and do MEG->mel.

        Args:
            x: (Optional) MEG data, shape [B, C, T].
            mel: (Optional) Mel data, shape [B, 80, T_mel].
            max_new_tokens: generation parameter.
            attention_mask: optional mask for the mel input.
            return_hidden_outputs: if True, returns the brain_module intermediate states.
            **gen_kwargs: additional generation config (e.g. temperature)
        Returns:
            token_ids: [batch_size, generated_sequence_length]
        """
        if mel is not None:
            if mel.size(1) != 80:
                raise ValueError(f"mel must be [B, 80, T], got {mel.size(1)} channels")
            input_features = self.pad_truncate(mel, max_frames=3000)
            quantizer_metrics, channel_weights, hidden_outputs = None, None, None
        elif x is not None:
            predicted_mel, quantizer_metrics, channel_weights, hidden_outputs = (
                self.brain_module(
                    x,
                    recording,
                    conditions,
                    mel,
                    train=False,
                    return_hidden_outputs=return_hidden_outputs,
                )
            )

            input_features = self.pad_truncate(predicted_mel, max_frames=3000)
        else:
            raise ValueError("Please provide either `x` (MEG) or `mel` to generate.")

        token_ids = self.decoder.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **gen_kwargs,
        )
        return (
            token_ids,
            input_features,
            quantizer_metrics,
            channel_weights,
            hidden_outputs,
        )

    def pad_truncate(
        self, tensor: torch.Tensor, max_frames: int = 3000
    ) -> torch.Tensor:
        B, C, T = tensor.size()
        if T < max_frames:
            pad_len = max_frames - T
            return nn.functional.pad(tensor, (0, pad_len), mode="constant", value=-0.2)
        elif T > max_frames:
            return tensor[:, :, :max_frames]
        return tensor

    def match_modules_string(
        self,
        named_modules,
        start_prefixes,
        end_suffixes,
        mid_prefixes=[],
    ):
        """
        Helper to find modules matching LoRA injection criteria,
        e.g. "model.encoder.layers.*.self_attn.k_proj" etc.
        """
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
