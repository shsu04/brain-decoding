import shutil
import typing as tp
import gc
import random
import time
from tqdm import tqdm
import json
from torch.optim import AdamW
import os
import torch
from transformers import WhisperForConditionalGeneration
import torch.nn as nn
from peft import PeftModel, PeftConfig

from dataloader import DataLoader
from dataloader.audio_text_batch import AudioTextBatch
from config import TrainingConfigV2
from losses.mse import mse_loss_per_batch
from losses.cos_sim import cosine_similarity_loss
from losses.mmd import MMDLoss
from train.training_session import TrainingSession
from models.whisper_decoder import WhisperDecoder

device = "cuda"


class TrainingSessionV2(TrainingSession):
    def __init__(
        self,
        config: TrainingConfigV2 = None,
        studies: tp.Dict[str, str] = None,
        data_path: str = "/home/ubuntu/brain-decoding/data",
        save_path: str = "/home/ubuntu/brain-decoding/saves",
        clear_cache: bool = False,
        max_cache_size: int = 100,
        cache_name: str = "cache",
        download_studies: bool = False,
    ):
        super().__init__(
            config=config,
            studies=studies,
            data_path=data_path,
            save_path=save_path,
            clear_cache=clear_cache,
            cache_enabled=True,
            max_cache_size=max_cache_size,
            cache_name=cache_name,
            download_studies=download_studies,
        )

        # MODEL
        self.model = WhisperDecoder(
            brain_module_config=config.brain_encoder_config,
            adalora_config=config.adalora_config,
            audio_model_id=config.audio_model,
        )

        if torch.cuda.is_available():
            self.optimizer = AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            self.scaler = torch.GradScaler(device=device)
        else:
            self.optimizer = None
            self.scaler = None
            print("CUDA is not available. Optimizer and scaler not initialized")

        (
            self.clip_loss_mel,
            self.clip_loss_latent,
            self.mse_loss,
            self.cosine_similarity_loss,
            self.mmd_loss,
        ) = (
            self.model.brain_module.clip_loss,
            self.model.clip_loss,
            mse_loss_per_batch,
            cosine_similarity_loss,
            MMDLoss(),
        )

        # Frozen whisper model for alignment
        self.frozen_encoder = WhisperForConditionalGeneration.from_pretrained(
            self.config.audio_model,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(device)
        self.frozen_encoder._freeze_parameters()

        self.adalora_steps = 0
        self.lowest_cer = float("inf")
        self.highest_bert = float("-inf")

        gpu_ok = False
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            if major >= 7:
                gpu_ok = True

        # (Optional) compile if on modern GPU
        if not gpu_ok:
            self.log_print(
                "GPU is not Ampere/Hopper, skipping torch.compile for speed."
            )
        if gpu_ok:
            self.frozen_encoder = torch.compile(
                self.frozen_encoder.forward,
                mode="reduce-overhead",
            )

        # Example scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=self.config.epochs * self.config.steps_per_epoch,
            pct_start=0.10,
            anneal_strategy="cos",
        )

    def train(
        self,
        device: str,
        buffer_size: int,
        num_workers: int,
        max_cache_size: int,
        current_epoch: int = 0,
    ):
        self.device = device
        self.model.to(device)
        self.clip_loss_mel.to(device)
        self.clip_loss_latent.to(device)
        self.mmd_loss.to(device)
        training_size = len(self.dataset["train"])

        for epoch in range(current_epoch + 1, self.config.epochs + 1):
            try:
                self.model.train()
                epoch_start_time = time.time()
                self.logger.info(f"Starting epoch {epoch}.")

                epoch_training_dataset = self.dataset["train"].copy()
                dataloader = self.get_dataloader(
                    buffer_size=buffer_size,
                    num_workers=num_workers,
                    max_cache_size=max_cache_size,
                    add_timestamps=self.config.decode_timestamps,
                    tokenize=True,
                )

                self.set_seed(int(self.config.seed + epoch))
                random.shuffle(epoch_training_dataset)
                dataloader.start_fetching(epoch_training_dataset, cache=True)

            except Exception as e:
                self.log_print(f"Error in epoch {epoch} during initialization, {e}")
                self.save(f"error_epoch_{epoch}")

            pbar = tqdm(
                total=len(epoch_training_dataset), desc="Training Epoch " + str(epoch)
            )
            all_metrics, total_batches = [], 0

            while True:
                batch = dataloader.get_recording()
                if batch is None:
                    break
                try:
                    results, num_batches = self.run_batch(batch, train=True)
                    all_metrics.append(results)
                    total_batches += num_batches
                except Exception as e:
                    self.log_print(
                        f"Error in epoch {epoch}, {batch.recording.study_name} {batch.recording.subject_id} {batch.recording.session_id} {batch.recording.task_id}. {e}"
                    )
                    self.save(f"error_epoch_{epoch}")
                    raise e

                del batch
                gc.collect()
                torch.cuda.empty_cache()
                pbar.update(1)

            dataloader.stop()
            pbar.close()

    def run_batch(self, batch: AudioTextBatch, train: bool) -> tp.Tuple[dict, int]:
        """
        The main place we do forward/backward.
        """
        (
            brain_segments,
            audio_segments,
            transcripts,
            transcript_attn_masks,
            recording,
        ) = (
            batch.brain_segments["all"],
            batch.audio_segments,
            batch.transcript,
            batch.transcript_attention_masks,
            batch.recording,
        )

        brain_segments, audio_segments, transcripts, transcript_attn_masks = (
            self.discard_nan(
                brain_segments, audio_segments, transcripts, transcript_attn_masks
            )
        )

    def test(self, buffer_size: int, num_workers: int, max_cache_size: int):
        pass

    def get_dataloader(
        self,
        buffer_size,
        num_workers,
        max_cache_size,
        add_timestamps: bool,
        tokenize: bool,
    ):
        dataloader = DataLoader(
            buffer_size=buffer_size,
            max_cache_size_gb=max_cache_size,
            cache_dir="cache",
            notch_filter=self.config.notch_filter,
            frequency_bands=self.config.frequency_bands,
            scaling=self.config.scaling,
            brain_clipping=self.config.brain_clipping,
            baseline_window=self.config.baseline_window,
            new_freq=self.config.new_freq,
            delay=self.config.delay,
            batch_types={"audiotext": num_workers},
            batch_kwargs={
                "audiotext": {
                    "max_random_shift": self.config.max_random_shift,
                    "window_size": self.config.window_size,
                    "window_stride": self.config.window_stride,
                    "audio_sample_rate": self.config.audio_sample_rate,
                    "hop_length": self.config.hop_length,
                    "audio_processor": self.config.audio_model,
                    "add_timestamps": add_timestamps,
                    "tokenize": tokenize,
                }
            },
        )
        return dataloader

    def discard_nan(
        self,
        brain: torch.Tensor,
        audio: torch.Tensor,
        transcript: torch.Tensor,
        attn_mask: torch.Tensor,
    ):
        """
        Arguments:
            brain -- [B, C, T] brain data
            audio -- [B, 80, T'] audio data
            transcript -- [B, T'] transcript data
            attn_mask -- [B, T'] attention mask
        """
        valid_mask = ~(
            torch.isnan(brain).any(dim=(1, 2))
            | torch.isnan(audio).any(dim=(1, 2))
            | torch.isnan(transcript).any(dim=(1))
            | torch.isnan(attn_mask).any(dim=(1))
        )

        if valid_mask.all():
            return brain, audio, transcript, attn_mask

        filtered_brain = brain[valid_mask]
        filtered_audio = audio[valid_mask]
        filtered_transcript = transcript[valid_mask]
        filtered_attn_mask = attn_mask[valid_mask]

        a, b, c, d = (
            filtered_brain.size(0),
            filtered_audio.size(0),
            filtered_transcript.size(0),
            filtered_attn_mask.size(0),
        )

        # Dim check for all 4
        assert (
            filtered_brain.size(0)
            == filtered_audio.size(0)
            == filtered_transcript.size(0)
            == filtered_attn_mask.size(0)
        ), f"Dimension mismatch! {a} {b} {c} {d}"

        return filtered_brain, filtered_audio, filtered_transcript, filtered_attn_mask

    def pre_process_all_recordings(
        self, buffer_size: int, num_workers: int, max_cache_size: int
    ):
        if self.recordings is None:
            self.partition_datasets()

        dataloader = self.get_dataloader(
            buffer_size,
            num_workers,
            max_cache_size,
            # Not needed for caching, done when retrieving
            add_timestamps=False,
            tokenize=False,
        )
        total_rec = len(self.recordings)
        pbar = tqdm(total=total_rec, desc="Loading recordings")

        dataloader.start_fetching(self.recordings, cache=True)
        done = 0
        while True:
            result = dataloader.get_recording()
            if result is None:
                break
            done += 1
            pbar.update(1)

    def save(self, name: str):
        pass


def load_training_session(
    save_path: str,
    studies: tp.Dict[str, str] = None,
    data_path: str = "/home/ubuntu/brain-decoding/data",
    clear_cache: bool = False,
    cache_name: str = None,
    max_cache_size: int = 100,
):
    pass
