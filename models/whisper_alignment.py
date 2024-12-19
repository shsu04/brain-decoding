import torch
import torch.nn as nn
from transformers import WhisperModel
from typing import List, Optional
from config import SimpleConvConfig
from models.simpleconv import SimpleConv


class WhisperAlignment(nn.Module):
    def __init__(
        self, 
        simpleconv_config: SimpleConvConfig,
        layers_to_align: Optional[List[int]] = [-1],
        use_compile: bool = False,
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    ):
        """Uses the encoder hidden states from a pre-trained model for alignment

        Arguments:
            simpleconv_config -- Brain encoder config

        Keyword Arguments:
            layers_to_align -- which hidden layers to output for alignment
            use_compile -- compile can have 4.5x speedup
        """
        super().__init__()
        self.simple_conv = SimpleConv(simpleconv_config)
        self.model_id = "openai/whisper-large-v3"
        
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        whisper_model = WhisperModel.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True
        ).to(device)
        
        # Only encoder is used for alignment, free mem
        self.encoder = whisper_model.get_encoder()
        del whisper_model.decoder
        del whisper_model

        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Which hidden layers to align, last by default
        self.layers_to_align = layers_to_align
        assert all([i < 32 for i in layers_to_align]), "Invalid layer index"

        if use_compile:
            self.compile()
            
        self.device = device
        self.to(device)
        self.half() if torch.cuda.is_available() else self.float()
        
    def compile(self):
        """Only used when inference is done"""
        self.encoder.forward = torch.compile(self.encoder.forward, mode="reduce-overhead", fullgraph=True)

    def forward(self, x: torch.Tensor) -> tuple[List[torch.Tensor], torch.Tensor]:
        """
        x: [B, C, T]
        
        Returns:
            List of hidden states for each layer in layers_to_align [B, 1500, 1280]
            Last hidden state [B, 1500, 1280]
        
            Where 1500 = T, 1280 = D
        """
        
        # x = self.simple_conv(x) # [B, 80, T]
        B, C, T = x.size()
        
        assert C == 128, f"Expected {128} channels, got {C}"
        assert T == 3000, f"Expected {3000} timesteps, got {T}"
        
        # Pad or truncate
        if T < 3000:
            pad_len = 3000 - T
            x = nn.functional.pad(x, (0, pad_len), mode='constant', value=0.0)
        elif T > 3000:
            # If longer, consider trimming or handling long-form audio differently.
            x = x[:, :, :3000]

        encoder_outputs = self.encoder(x, output_hidden_states=True)
        
        return ([
            encoder_outputs.hidden_states[i] for i in self.layers_to_align], 
            encoder_outputs.last_hidden_state
        )