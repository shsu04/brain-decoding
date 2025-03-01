import re
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
from transformers import (
    WhisperModel,
    WhisperTokenizerFast,
    WhisperForConditionalGeneration,
)
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
from utils.nlp_metrics import nlp_metrics


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
        frozen_whisper_model = WhisperModel.from_pretrained(
            config.audio_model,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(device)
        self.frozen_encoder = frozen_whisper_model.get_encoder()
        self.frozen_encoder._freeze_parameters()

        del frozen_whisper_model.decoder
        del frozen_whisper_model

        self.adalora_steps = 0
        self.lowest_cer = float("inf")
        self.highest_bert = float("-inf")

        gpu_ok = False
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            if major >= 8:
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

        # scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=self.config.epochs * self.config.steps_per_epoch,
            pct_start=0.10,
            anneal_strategy="cos",
        )

        # For testing, no timestamps.
        self.tokenizer = WhisperTokenizerFast.from_pretrained(
            self.config.audio_model,
            predict_timestamps=False,
            add_prefix_space=True,
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
                    results, num_batches = self.train_batch(batch, train=True)
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

            final_metrics = {
                key: sum([m[key] for m in all_metrics]) / training_size
                for key in all_metrics[0].keys()
            }
            self.metrics["train"].append(final_metrics)

            # Print
            self.log_no_print("\n")
            self.log_no_print(
                f'Epoch {epoch}, Loss: {final_metrics["loss"]:.4f}, CE Loss: {final_metrics["ce_loss"]:.4f}'
            )
            # Mel
            self.log_no_print(
                f"Mel Loss: {final_metrics['mel_loss']:.4f}, Clip Loss: {final_metrics['clip_loss']:.4f}, MSE: {final_metrics['mse_loss']:.4f}"
            )
            self.log_no_print(
                f"Mel Accuracy: {final_metrics['accuracy']:.4f}, Top 5: {final_metrics['top_5_accuracy']:.4f}, Top 10: {final_metrics['top_10_accuracy']:.4f}"
            )
            # Encoder
            self.log_no_print(
                f"Encoder Loss: {final_metrics['encoder_total_loss']:.4f}, Cos Sim: {final_metrics['encoder_cosine_similarity_loss']:.4f}, MSE: {final_metrics['encoder_mse_loss']:.4f}, Clip: {final_metrics['encoder_clip_loss']:.4f}, MMD: {final_metrics['encoder_mmd_loss']:.4f}"
            )
            self.log_no_print(
                f"Encoder Accuracy: {final_metrics['encoder_accuracy']:.4f}, Top 5: {final_metrics['encoder_top_5_accuracy']:.4f}, Top 10: {final_metrics['encoder_top_10_accuracy']:.4f}"
            )

            # Testing
            try:
                with torch.no_grad():
                    self.logger.info(f"Starting testing for epoch {epoch}.")
                    self.test(
                        buffer_size=buffer_size,
                        num_workers=num_workers,
                        max_cache_size=max_cache_size,
                    )
            except Exception as e:
                self.log_print(f"Error in epoch {epoch} during testing, {e}")
                self.save(f"error_epoch_{epoch}")
                raise e

            elapsed_minutes = (time.time() - epoch_start_time) / 60
            self.log_print(
                f"Epoch {epoch} done in {elapsed_minutes:.2f}m. {elapsed_minutes / training_size:.2f}m/recording."
            )
            self.save(f"epoch_{epoch}")

            # Early stopping logic
            average_cer = (
                sum(
                    self.metrics["test"][test][-1]["nlp_metrics"]["cer"]
                    for test in self.metrics["test"]
                )
                / 3
            )
            average_bert = (
                sum(
                    self.metrics["test"][test][-1]["nlp_metrics"]["bert_score"]
                    for test in self.metrics["test"]
                )
                / 3
            )

            if average_cer < self.lowest_cer and average_bert > self.highest_bert:
                self.lowest_cer = average_cer
                self.highest_bert = average_bert
                self.highest_epoch = epoch
                self.highest_metrics = {
                    test: self.metrics["test"][test][-1]
                    for test in self.metrics["test"].keys()
                }

                self.log_print(
                    f"New best epoch {epoch} with CER {average_cer:.4f} and BERT {average_bert:.4f}."
                )
                self.log_print(
                    f"Mel Loss: {final_metrics['mel_loss']:.4f}, Clip Loss: {final_metrics['clip_loss']:.4f}, MSE: {final_metrics['mse_loss']:.4f}"
                )
                self.log_print(
                    f"Mel accuracy: {final_metrics['accuracy']:.4f}, Top 5: {final_metrics['top_5_accuracy']:.4f}, Top 10: {final_metrics['top_10_accuracy']:.4f}"
                )

            if epoch - self.highest_epoch > 10:
                self.log_print(
                    f"Early stopping at epoch {epoch}. Highest metrics at epoch {self.highest_epoch}."
                )
                break

        self.log_print("\n")
        self.log_print(f"Training completed. Highest epoch at {self.highest_epoch}.")

        for test, metric in self.highest_metrics.items():
            self.log_print("\n")
            self.log_print(
                f"Test {test} at epoch {self.highest_epoch}. Mel Loss: {metric['mel_loss']:.4f}, Clip Loss: {metric['clip_loss']:.4f}, MSE: {metric['mse_loss']:.4f}"
            )
            self.log_print(
                f"Mel accuracy: {metric['accuracy']:.4f}, Top 5: {metric['top_5_accuracy']:.4f}, Top 10: {metric['top_10_accuracy']:.4f}"
            )
            self.log_print(
                f"BLEU: {metric['nlp_metrics']['bleu']:.4f}, ROUGE-1: {metric['nlp_metrics']['rouge-f']:.4f}, BERT: {metric['nlp_metrics']['bert_score']:.4f}, CER: {metric['nlp_metrics']['cer']:.4f}, SELF-BLEU: {metric['nlp_metrics']['self_bleu']:.4f}"
            )

    def train_batch(self, batch: AudioTextBatch, train: bool) -> tp.Tuple[dict, int]:
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

        # Accumulators for mel and total losses
        recording_loss = 0.0
        recording_mel_loss = 0.0
        recording_clip_loss = 0.0
        recording_mse_loss = 0.0

        # Encoder Alignment losses
        recording_encoder_cosine_similarity_losses = 0.0
        recording_encoder_mse_losses = 0.0
        recording_encoder_clip_losses = 0.0
        recording_encoder_mmd_losses = 0.0
        recording_encoder_total_losses = 0.0

        # Decoder losses
        recording_ce_loss = 0.0

        # For mel-level clip correctness
        recording_correct = 0.0
        recording_top_5 = 0.0
        recording_top_10 = 0.0

        # For latent-level clip correctness
        recording_encoder_correct = 0.0
        recording_encoder_top5 = 0.0
        recording_encoder_top10 = 0.0

        # To correctly average the losses
        total_samples = brain_segments.shape[0]
        missed_recordings = 0
        missed_batches = 0

        conditions = {
            "study": f"{recording.study_name}",
            "subject": f"{recording.study_name}_{recording.subject_id}",
        }

        # Random shuffle
        shuffle_idx = torch.randperm(total_samples)
        brain_segments = brain_segments[shuffle_idx]  # [B, C, T]
        audio_segments = audio_segments[shuffle_idx]  # [B, 80, T']
        transcripts = transcripts[shuffle_idx]  # [B, T']
        transcript_attn_masks = transcript_attn_masks[shuffle_idx]  # [B, T']

        # For slicing
        batch_indices = [
            (i, min(i + self.config.batch_size, total_samples))
            for i in range(0, total_samples, self.config.batch_size)
        ]

        with torch.autocast(device_type=device, dtype=self.autocast_dtype):
            for start, end in batch_indices:
                try:
                    if train:
                        self.optimizer.zero_grad()

                    brain_batch = brain_segments[start:end].to(device)
                    audio_batch = audio_segments[start:end].to(device)
                    transcript_batch = transcripts[start:end].to(device)
                    transcript_attn_mask_batch = transcript_attn_masks[start:end].to(
                        device
                    )

                    # Encoder attention mask
                    pad_len = 3000 - audio_batch.size(2)
                    encoder_attention_mask = torch.zeros(
                        audio_batch.size(0), 3000, device=device
                    )  # [B, T]
                    encoder_attention_mask[:, : audio_batch.size(2)] = 1

                    (
                        x,  # predicted mel
                        _,
                        _,
                        _,
                        encoder_last_hidden_state,
                        ce_loss,
                    ) = self.model(
                        x=[brain_batch],
                        recording=[recording],
                        conditions=[conditions],
                        mel=[audio_batch],
                        train=train,
                        # Set true using this function only when testing in isolation
                        return_hidden_outputs=False,
                        attention_mask=encoder_attention_mask,
                        labels=transcript_batch,
                        decoder_attention_mask=transcript_attn_mask_batch,
                    )

                    # Frozen encoder
                    with torch.no_grad():
                        outputs = self.frozen_encoder(
                            nn.functional.pad(
                                audio_batch,
                                (0, pad_len),
                                mode="constant",
                                value=-0.2,
                            ),
                            output_hidden_states=False,
                            attention_mask=encoder_attention_mask,
                        )
                        # [B, T, D]
                        frozen_encoder_last_hidden_state = outputs.last_hidden_state[
                            :, : audio_batch.size(2), :
                        ]
                        del outputs, encoder_attention_mask, transcript_attn_mask_batch
                        gc.collect()

                    # Mel alignment objectives
                    if self.config.mel_alignment_objectives["mse_loss"] > 0:
                        mse_l = self.mse_loss(pred=x, target=audio_batch)
                    else:
                        mse_l = torch.tensor(0.0).to(device)

                    if self.config.mel_alignment_objectives["clip_loss"] > 0:
                        clip_mel = self.clip_loss_mel(x_1=x, x_2=audio_batch, mel=True)
                        mel_clip_loss, mel_clip_metrics = (
                            clip_mel["loss"],
                            clip_mel["metrics"],
                        )
                    else:
                        mel_clip_loss = torch.tensor(0.0).to(device)
                        mel_clip_metrics = {
                            "correct": 0,
                            "top_5_correct": 0,
                            "top_10_correct": 0,
                        }

                    mel_loss = (
                        self.config.mel_alignment_objectives["mse_loss"] * mse_l
                        + self.config.mel_alignment_objectives["clip_loss"]
                        * mel_clip_loss
                    )

                    # Encoder alignment objectives
                    if self.config.latent_alignment_objectives["cosine_similarity"] > 0:
                        encoder_cos_sim_loss = cosine_similarity_loss(
                            frozen_encoder_last_hidden_state, encoder_last_hidden_state
                        )
                    else:
                        encoder_cos_sim_loss = torch.tensor(0.0).to(device)

                    if self.config.latent_alignment_objectives["mse_loss"] > 0:
                        encoder_mse_loss = mse_loss_per_batch(
                            frozen_encoder_last_hidden_state, encoder_last_hidden_state
                        )
                    else:
                        encoder_mse_loss = torch.tensor(0.0).to(device)

                    if self.config.latent_alignment_objectives["clip_loss"] > 0:
                        # [B, T, D] -> [B, D, T]
                        encoder_clip_loss = self.clip_loss_latent(
                            x_1=encoder_last_hidden_state.transpose(1, 2),
                            x_2=frozen_encoder_last_hidden_state.transpose(1, 2),
                            mel=False,
                        )
                        encoder_clip_loss, encoder_clip_metrics = (
                            encoder_clip_loss["loss"],
                            encoder_clip_loss["metrics"],
                        )
                    else:
                        encoder_clip_loss = torch.tensor(0.0).to(device)
                        encoder_clip_metrics = {
                            "correct": 0,
                            "top_5_correct": 0,
                            "top_10_correct": 0,
                        }

                    if self.config.latent_alignment_objectives["mmd_loss"] > 0:
                        encoder_mmd_loss = self.mmd_loss(
                            encoder_last_hidden_state, frozen_encoder_last_hidden_state
                        )
                    else:
                        encoder_mmd_loss = torch.tensor(0.0).to(device)

                    encoder_loss = (
                        self.config.latent_alignment_objectives["cosine_similarity"]
                        * encoder_cos_sim_loss
                        + self.config.latent_alignment_objectives["mse_loss"]
                        * encoder_mse_loss
                        + self.config.latent_alignment_objectives["clip_loss"]
                        * encoder_clip_loss
                        + self.config.latent_alignment_objectives["mmd_loss"]
                        * encoder_mmd_loss
                    )

                    total_loss = mel_loss + encoder_loss + ce_loss

                    # Optimize
                    if not torch.isnan(total_loss).any():
                        if train:
                            self.scaler.scale(total_loss).backward()

                            # Clip gradients
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), max_norm=3.0
                            )

                            if self.adalora_steps >= self.config.adalora_config.tinit:
                                self.model.decoder.base_model.update_and_allocate(
                                    self.adalora_steps
                                )
                            if self.adalora_steps == self.config.adalora_config.tinit:
                                self.log_print(
                                    f"Starting rank reallocation at recording {self.adalora_steps}."
                                )

                            # Step
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.optimizer.zero_grad()

                            self.adalora_steps += 1
                            self.scheduler.step()

                        # Accumulate losses
                        recording_mel_loss += mel_loss.detach().cpu().item()
                        recording_clip_loss += mel_clip_loss.detach().cpu().item()
                        recording_mse_loss += mse_l.detach().cpu().item()

                        recording_encoder_cosine_similarity_losses += (
                            encoder_cos_sim_loss.detach().cpu().item()
                        )
                        recording_encoder_mse_losses += (
                            encoder_mse_loss.detach().cpu().item()
                        )
                        recording_encoder_clip_losses += (
                            encoder_clip_loss.detach().cpu().item()
                        )
                        recording_encoder_mmd_losses += (
                            encoder_mmd_loss.detach().cpu().item()
                        )
                        recording_encoder_total_losses += (
                            encoder_loss.detach().cpu().item()
                        )

                        recording_ce_loss += ce_loss.detach().cpu().item()
                        recording_loss += total_loss.detach().cpu().item()

                        # Accumulate correctness
                        recording_correct += mel_clip_metrics["correct"]
                        recording_top_5 += mel_clip_metrics["top_5_correct"]
                        recording_top_10 += mel_clip_metrics["top_10_correct"]

                        recording_encoder_correct += encoder_clip_metrics["correct"]
                        recording_encoder_top5 += encoder_clip_metrics["top_5_correct"]
                        recording_encoder_top10 += encoder_clip_metrics[
                            "top_10_correct"
                        ]

                    else:
                        self.log_print(
                            f"NaN loss on {recording.study_name} subj {recording.subject_id} task {recording.task_id}."
                        )
                        missed_recordings += end - start
                        missed_batches += 1

                except Exception as ex:
                    self.log_print(f"Error in train_batch: {ex}")
                    missed_recordings += end - start
                    missed_batches += 1
                    raise ex

        # End of batch loop
        total_samples -= missed_recordings
        batches = len(batch_indices) - missed_batches

        metrics = {
            "loss": recording_loss / batches if batches > 0 else 0.0,
            "mel_loss": recording_mel_loss / batches if batches > 0 else 0.0,
            "clip_loss": recording_clip_loss / batches if batches > 0 else 0.0,
            "mse_loss": recording_mse_loss / batches if batches > 0 else 0.0,
            # Encoder alignment losses
            "encoder_cosine_similarity_loss": (
                recording_encoder_cosine_similarity_losses / batches
                if batches > 0
                else 0.0
            ),
            "encoder_mse_loss": (
                recording_encoder_mse_losses / batches if batches > 0 else 0.0
            ),
            "encoder_clip_loss": (
                recording_encoder_clip_losses / batches if batches > 0 else 0.0
            ),
            "encoder_mmd_loss": (
                recording_encoder_mmd_losses / batches if batches > 0 else 0.0
            ),
            "encoder_total_loss": (
                recording_encoder_total_losses / batches if batches > 0 else 0.0
            ),
            # Decoder losses
            "ce_loss": recording_ce_loss / batches if batches > 0 else 0.0,
            # Accuracy
            "accuracy": recording_correct / total_samples if total_samples > 0 else 0.0,
            "top_5_accuracy": (
                recording_top_5 / total_samples if total_samples > 0 else 0.0
            ),
            "top_10_accuracy": (
                recording_top_10 / total_samples if total_samples > 0 else 0.0
            ),
            # Encoder accuracy
            "encoder_accuracy": (
                recording_encoder_correct / total_samples if total_samples > 0 else 0.0
            ),
            "encoder_top_5_accuracy": (
                recording_encoder_top5 / total_samples if total_samples > 0 else 0.0
            ),
            "encoder_top_10_accuracy": (
                recording_encoder_top10 / total_samples if total_samples > 0 else 0.0
            ),
        }

        self.logger.info(
            f"{recording.study_name} {recording.subject_id} sess {recording.session_id}, Loss: {metrics['loss']:.4f}, CE Loss: {metrics['ce_loss']:.4f}"
        )
        # Mel
        self.logger.info(
            f"Mel Loss: {metrics['mel_loss']:.4f}, Clip Loss: {metrics['clip_loss']:.4f}, MSE: {metrics['mse_loss']:.4f}"
        )
        self.logger.info(
            f"Mel Accuracy: {metrics['accuracy']:.4f}, Top 5: {metrics['top_5_accuracy']:.4f}, Top 10: {metrics['top_10_accuracy']:.4f}"
        )
        # Encoder
        self.logger.info(
            f"Encoder Loss: {metrics['encoder_total_loss']:.4f}, Cos Sim: {metrics['encoder_cosine_similarity_loss']:.4f}, MSE: {metrics['encoder_mse_loss']:.4f}, Clip: {metrics['encoder_clip_loss']:.4f}, MMD: {metrics['encoder_mmd_loss']:.4f}"
        )
        self.logger.info(
            f"Encoder Accuracy: {metrics['encoder_accuracy']:.4f}, Top 5: {metrics['encoder_top_5_accuracy']:.4f}, Top 10: {metrics['encoder_top_10_accuracy']:.4f}"
        )
        gc.collect()
        torch.cuda.empty_cache()

        return metrics, total_samples

    def test(self, buffer_size: int, num_workers: int, max_cache_size: int):
        self.model.eval().to(self.device)
        self.set_seed(int(self.config.seed))
        start_time = time.time()

        test_datasets, test_sizes, test_dataloader = {}, {}, {}

        # Gather test set and loader
        for test in self.dataset["test"].keys():

            if len(self.dataset["test"][test]) < self.config.random_test_size:
                test_datasets[test] = self.dataset["test"][test]
            else:
                test_datasets[test] = random.sample(
                    self.dataset["test"][test], self.config.random_test_size
                )

            test_sizes[test] = len(test_datasets[test])
            test_dataloader[test] = self.get_dataloader(
                buffer_size=test_sizes[test],
                num_workers=test_sizes[test],
                max_cache_size=max_cache_size,
                # NLP eval without timestamps
                add_timestamps=False,
                tokenize=False,
            )
            test_dataloader[test].start_fetching(test_datasets[test], cache=True)

        # Test loop
        with torch.no_grad():
            for test in test_datasets:
                all_metrics, total_batches = [], 0
                while True:
                    batch = test_dataloader[test].get_recording()
                    if batch is None:
                        break
                    try:
                        results, num_batches = self.test_batch(batch)
                        all_metrics.append(results)
                        total_batches += num_batches
                    except Exception as e:
                        self.log_print(f"Error in testing {test}, skipping. {e}")
                        test_sizes[test] -= 1
                        continue
                    del batch
                    gc.collect()

                if test_sizes[test] == 0:
                    continue

                final_metrics = {
                    key: sum([m[key] for m in all_metrics]) / test_sizes[test]
                    for key in all_metrics[0].keys()
                }
                self.metrics["test"][test].append(final_metrics)

                self.log_no_print("\n")
                self.log_no_print(
                    f'Test {test} done. Mel Loss: {final_metrics["mel_loss"]:.4f}, Clip Loss: {final_metrics["clip_loss"]:.4f}, MSE: {final_metrics["mse_loss"]:.4f}'
                )
                self.log_no_print(
                    f"Mel Accuracy: {final_metrics['accuracy']:.4f}, Top 5: {final_metrics['top_5_accuracy']:.4f}, Top 10: {final_metrics['top_10_accuracy']:.4f}"
                )
                self.log_no_print(
                    f"BLEU: {final_metrics['nlp_metrics']['bleu']:.4f}, ROUGE-1: {final_metrics['nlp_metrics']['rouge-f']:.4f}, BERT: {final_metrics['nlp_metrics']['bert_score']:.4f}, CER: {final_metrics['nlp_metrics']['cer']:.4f}, SELF-BLEU: {final_metrics['nlp_metrics']['self_bleu']:.4f}"
                )

                test_dataloader[test].stop()

        elaps = (time.time() - start_time) / 60
        self.log_print(f"Testing done in {elaps:.2f}m.")

    def test_batch(self, batch: AudioTextBatch) -> tp.Tuple[dict, int]:
        """
        New function for testing batch, eval only on mel-level and NLP metrics
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

        # Accumulators for mel and total losses
        recording_mel_loss = 0.0
        recording_clip_loss = 0.0
        recording_mse_loss = 0.0

        # For mel-level clip correctness
        recording_correct = 0.0
        recording_top_5 = 0.0
        recording_top_10 = 0.0

        # For NLP metrics
        recording_nlp_metrics = []

        # To correctly average the losses
        total_samples = brain_segments.shape[0]
        missed_recordings = 0
        missed_batches = 0

        conditions = {
            "study": f"{recording.study_name}",
            "subject": f"{recording.study_name}_{recording.subject_id}",
        }

        # Random shuffle
        shuffle_idx = torch.randperm(total_samples)
        brain_segments = brain_segments[shuffle_idx]  # [B, C, T]
        audio_segments = audio_segments[shuffle_idx]  # [B, 80, T']
        transcripts = transcripts[shuffle_idx]  # [B, T']
        transcript_attn_masks = transcript_attn_masks[shuffle_idx]  # [B, T']

        # For slicing
        batch_indices = [
            (i, min(i + self.config.batch_size, total_samples))
            for i in range(0, total_samples, self.config.batch_size)
        ]

        with torch.autocast(device_type=device, dtype=self.autocast_dtype):
            for start, end in batch_indices:
                try:
                    brain_batch = brain_segments[start:end].to(device)
                    audio_batch = audio_segments[start:end].to(device)
                    transcript_batch = transcripts[start:end].to(device)
                    transcript_attn_mask_batch = transcript_attn_masks[start:end].to(
                        device
                    )
                    # Encoder attention mask
                    pad_len = 3000 - audio_batch.size(2)
                    encoder_attention_mask = torch.zeros(
                        audio_batch.size(0), 3000, device=device
                    )  # [B, T]
                    encoder_attention_mask[:, : audio_batch.size(2)] = 1

                    # Model generate
                    (
                        token_ids,  # [B, T]
                        x,  # [B, 80, T']
                        quantizer_metrics,
                        channel_weights,
                        hidden_outputs,
                    ) = self.model.generate(
                        x=[brain_batch],
                        recording=[recording],
                        conditions=[conditions],
                        mel=None,
                        max_new_tokens=128,
                        attention_mask=encoder_attention_mask,
                        return_hidden_outputs=False,
                    )
                    del (
                        quantizer_metrics,
                        encoder_attention_mask,
                        transcript_attn_mask_batch,
                        channel_weights,
                        hidden_outputs,
                    )
                    gc.collect()

                    # Mel alignment objectives
                    if self.config.mel_alignment_objectives["mse_loss"] > 0:
                        mse_l = self.mse_loss(pred=x, target=audio_batch)
                    else:
                        mse_l = torch.tensor(0.0).to(device)

                    if self.config.mel_alignment_objectives["clip_loss"] > 0:
                        clip_mel = self.clip_loss_mel(x_1=x, x_2=audio_batch, mel=True)
                        mel_clip_loss, mel_clip_metrics = (
                            clip_mel["loss"],
                            clip_mel["metrics"],
                        )
                    else:
                        mel_clip_loss = torch.tensor(0.0).to(device)
                        mel_clip_metrics = {
                            "correct": 0,
                            "top_5_correct": 0,
                            "top_10_correct": 0,
                        }

                    mel_loss = (
                        self.config.mel_alignment_objectives["mse_loss"] * mse_l
                        + self.config.mel_alignment_objectives["clip_loss"]
                        * mel_clip_loss
                    )

                    # Decode and clean up
                    token_ids = self.tokenizer.batch_decode(
                        sequences=token_ids,
                        skip_special_tokens=True,
                        decode_with_timestamps=False,
                        clean_up_tokenization_spaces=True,
                    )

                    # Accumulate Mel
                    recording_mel_loss += mel_loss.detach().cpu().item()
                    recording_clip_loss += mel_clip_loss.detach().cpu().item()
                    recording_mse_loss += mse_l.detach().cpu().item()
                    recording_correct += mel_clip_metrics["correct"]
                    recording_top_5 += mel_clip_metrics["top_5_correct"]
                    recording_top_10 += mel_clip_metrics["top_10_correct"]

                    # Accumulate NLP metrics
                    recording_nlp_metrics.append(
                        nlp_metrics(
                            predictions=token_ids,
                            references=transcript_batch,
                            batch_size=self.config.batch_size,
                        )
                    )

                except Exception as ex:
                    self.log_print(f"Error in test_batch: {ex}")
                    missed_recordings += end - start
                    missed_batches += 1
                    raise ex

        # End of batch loop
        total_samples -= missed_recordings
        batches = len(batch_indices) - missed_batches

        metrics = {
            "mel_loss": recording_mel_loss / batches if batches > 0 else 0.0,
            "clip_loss": recording_clip_loss / batches if batches > 0 else 0.0,
            "mse_loss": recording_mse_loss / batches if batches > 0 else 0.0,
            # Accuracy
            "accuracy": recording_correct / total_samples if total_samples > 0 else 0.0,
            "top_5_accuracy": (
                recording_top_5 / total_samples if total_samples > 0 else 0.0
            ),
            "top_10_accuracy": (
                recording_top_10 / total_samples if total_samples > 0 else 0.0
            ),
            # NLP metrics
            "nlp_metrics": {
                key: sum([m[key] for m in recording_nlp_metrics]) / batches
                for key in recording_nlp_metrics[0].keys()
            },
        }

        self.logger.info(
            f"Test {recording.study_name} {recording.subject_id} sess {recording.session_id}"
        )
        self.logger.info(
            f"Mel Loss: {metrics['mel_loss']:.4f}, Clip Loss: {metrics['clip_loss']:.4f}, MSE: {metrics['mse_loss']:.4f}"
        )
        self.logger.info(
            f"Mel Accuracy: {metrics['accuracy']:.4f}, Top 5: {metrics['top_5_accuracy']:.4f}, Top 10: {metrics['top_10_accuracy']:.4f}"
        )
        self.logger.info(
            f"BLEU: {metrics['nlp_metrics']['bleu']:.4f}, ROUGE-1: {metrics['nlp_metrics']['rouge-f']:.4f}, BERT: {metrics['nlp_metrics']['bert_score']:.4f}, CER: {metrics['nlp_metrics']['cer']:.4f}, SELF-BLEU: {metrics['nlp_metrics']['self_bleu']:.4f}"
        )

        gc.collect()
        torch.cuda.empty_cache()

        return metrics, total_samples

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
        with torch.no_grad():

            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

            config = self.config.to_dict()
            with open(self.save_path + "/training_config.json", "w") as jf:
                json.dump(config, jf, indent=4)
            ckp_path = f"{self.save_path}/{name}"
            os.makedirs(ckp_path, exist_ok=True)

            # Save model
            torch.save(
                {
                    "config": self.config.to_dict(),
                    "brain_encoder": self.model.brain_module.cpu().state_dict(),
                    "conditions": self.model.brain_module.condition_to_idx,
                },
                f"{ckp_path}/brain_encoder.pt",
            )
            self.model.decoder.save_pretrained(f"{ckp_path}/adalora_adapter")

            # Save metrics
            torch.save(
                {
                    "metrics": self.metrics,
                    "error": str(self.error) if self.error else "No errors.",
                    "highest_epoch": self.highest_epoch,
                    "highest_metrics": self.highest_metrics,
                    "lowest_cer": self.lowest_cer,
                    "highest_bert": self.highest_bert,
                    "optimizer": self.optimizer.state_dict(),
                    "adalora_steps": self.adalora_steps,
                    "scaler": self.scaler.state_dict(),
                },
                f"{ckp_path}/metrics.pt",
            )

        self.model.to(self.device)
        gc.collect()
        torch.cuda.empty_cache()
        return


def load_training_session(
    save_path: str,
    studies: tp.Dict[str, str] = None,
    data_path: str = "/home/ubuntu/brain-decoding/data",
    clear_cache: bool = False,
    cache_name: str = None,
    max_cache_size: int = 100,
):
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Save path {save_path} does not exist.")

    brain_ckp_path = os.path.join(save_path, "brain_encoder.pt")
    if not os.path.exists(brain_ckp_path):
        raise ValueError(f"Cannot find {brain_ckp_path}.")
    brain_ckp = torch.load(brain_ckp_path, map_location="cpu")
    config_dict = brain_ckp["config"]
    config = TrainingConfigV2(
        brain_encoder_config=None,
        data_partition=None,
    ).from_dict(config_dict)

    ts = TrainingSessionV2(
        config=config,
        studies=studies,
        data_path=data_path,
        save_path="temp",
        clear_cache=clear_cache,
        cache_name=cache_name,
        max_cache_size=max_cache_size,
    )
    ts.save_path = save_path

    ts.model.brain_module.load_state_dict(brain_ckp["brain_encoder"])
    if ts.model.brain_module.condition_to_idx != brain_ckp["conditions"]:
        raise ValueError("Condition mismatch.")

    adalora_adapter_path = os.path.join(save_path, "adalora_adapter")
    if not os.path.exists(adalora_adapter_path):
        raise ValueError(f"No adalora_adapter in {adalora_adapter_path}.")

    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        config.audio_model,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)

    peft_enc = PeftModel.from_pretrained(whisper_model, adalora_adapter_path)
    adalora_config = PeftConfig.from_pretrained(adalora_adapter_path)
    ts.model.adalora_config = adalora_config
    ts.model.decoder = peft_enc

    metrics_path = os.path.join(save_path, "metrics.pt")
    if os.path.exists(metrics_path):
        loaded = torch.load(metrics_path, map_location="cpu")
        if "metrics" in loaded:
            ts.metrics = loaded["metrics"]
        else:
            ts.metrics = {}
        ts.highest_epoch = loaded.get("highest_epoch", 0)
        ts.highest_metrics = loaded.get("highest_metrics", {})

        ts.lowest_cer = loaded.get("lowest_cer", float("inf"))
        ts.highest_bert = loaded.get("highest_bert", float("-inf"))
        ts.adalora_steps = loaded.get("adalora_steps", 0)

        if "optimizer" in loaded:
            ts.optimizer.load_state_dict(loaded["optimizer"])
        if "scaler" in loaded:
            ts.scaler.load_state_dict(loaded["scaler"])
        ts.error = loaded.get("error", None)
    else:
        ts.metrics = {}
        ts.logger.warning(f"No metrics found at {metrics_path}.")

    shutil.rmtree("temp")
    gc.collect()
    torch.cuda.empty_cache()
    return ts
