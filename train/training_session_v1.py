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
from transformers import WhisperModel
import torch.nn as nn
from peft import PeftModel, PeftConfig

from dataloader import DataLoader
from dataloader.audio_batch import AudioBatch
from config import TrainingConfigV1
from losses.mse import mse_loss_per_batch
from losses.cos_sim import cosine_similarity_loss
from train.training_session import TrainingSession
from models.whisper_alignment import WhisperAlignment

device = "cuda"


class TrainingSessionV1(TrainingSession):
    def __init__(
        self,
        config: TrainingConfigV1 = None,
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
        self.model = WhisperAlignment(
            brain_module_config=config.brain_encoder_config,
            adalora_config=config.adalora_config,
            layers_to_align=config.latent_alignment_layers,
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
        ) = (
            self.model.brain_module.clip_loss,
            self.model.clip_loss,
            mse_loss_per_batch,
            cosine_similarity_loss,
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

        torch.cuda.empty_cache()
        gc.collect()

        self.adalora_steps = 0
        self.lowest_final_layer_total_loss = float("inf")

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
                self.frozen_encoder.forward, mode="default"
            )

        # Example scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=self.config.epochs * len(self.dataset["train"]),
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
                pbar.update(1)

            pbar.close()

            # Aggregate metrics across the entire epoch
            final_metrics = {
                "loss": sum(b["loss"] for b in all_metrics) / training_size,
                "mel_loss": sum(b["mel_loss"] for b in all_metrics) / training_size,
                "clip_loss": sum(b["clip_loss"] for b in all_metrics) / training_size,
                "cosine_similarity_loss": sum(
                    b["cosine_similarity_loss"] for b in all_metrics
                )
                / training_size,
                "mse_loss": sum(b["mse_loss"] for b in all_metrics) / training_size,
                "commitment_loss": sum(b["commitment_loss"] for b in all_metrics)
                / training_size,
                "perplexity": sum(b["perplexity"] for b in all_metrics) / training_size,
                "alignment_losses": {
                    key: [
                        sum(b["alignment_losses"][key][i] for b in all_metrics)
                        / training_size
                        for i in range(len(self.config.latent_alignment_layers))
                    ]
                    for key in all_metrics[0]["alignment_losses"]
                },
                "final_layer_losses": {
                    key: sum(b["final_layer_losses"][key] for b in all_metrics)
                    / training_size
                    for key in all_metrics[0]["final_layer_losses"]
                },
                "latent_alignment_metrics": {},
                "accuracy": sum(b["accuracy"] for b in all_metrics) / training_size,
                "top_5_accuracy": sum(b["top_5_accuracy"] for b in all_metrics)
                / training_size,
                "top_10_accuracy": sum(b["top_10_accuracy"] for b in all_metrics)
                / training_size,
            }

            # If we have latent alignment metrics, average them across the entire epoch
            if "latent_alignment_metrics" in all_metrics[0]:
                lam_keys = all_metrics[0]["latent_alignment_metrics"].keys()
                final_lam: tp.Dict[str, tp.List[float]] = {}
                for lam_key in lam_keys:
                    layer_vals = []
                    for layer_idx in range(len(self.config.latent_alignment_layers)):
                        total_val = 0.0
                        for m in all_metrics:
                            total_val += m["latent_alignment_metrics"][lam_key][
                                layer_idx
                            ]
                        layer_vals.append(total_val / training_size)
                    final_lam[lam_key] = layer_vals
                final_metrics["latent_alignment_metrics"] = final_lam

            self.metrics["train"].append(final_metrics)

            # Print
            self.log_no_print("\n")
            self.log_no_print(
                f"Epoch {epoch}, Loss: {final_metrics['loss']:.4f}, Mel Loss: {final_metrics['mel_loss']:.4f}"
            )
            self.log_no_print(
                f"Clip Loss: {final_metrics['clip_loss']:.4f}, MSE Loss: {final_metrics['mse_loss']:.4f}, CosSim: {final_metrics['cosine_similarity_loss']:.4f}, Commit: {final_metrics['commitment_loss']:.4f}"
            )
            self.log_no_print(
                f"Perplexity: {final_metrics['perplexity']:.4f}, Acc: {final_metrics['accuracy']:.4f}, Top5: {final_metrics['top_5_accuracy']:.4f}, Top10: {final_metrics['top_10_accuracy']:.4f}"
            )

            # Print final-layer alignment metrics if available
            if "latent_alignment_metrics" in final_metrics:
                lam = final_metrics["latent_alignment_metrics"]
                # Show final layer clip accuracy, top5, top10
                if "clip_accuracy" in lam:
                    last_l = len(lam["clip_accuracy"]) - 1
                    final_acc = lam["clip_accuracy"][last_l]
                    final_acc_5 = lam["clip_top5_accuracy"][last_l]
                    final_acc_10 = lam["clip_top10_accuracy"][last_l]
                    self.log_no_print(
                        f"Final layer latent clip acc: {final_acc:.4f}, top5: {final_acc_5:.4f}, top10: {final_acc_10:.4f}"
                    )

            fll = final_metrics["final_layer_losses"]
            self.log_no_print(
                f"FinLayer Clip Loss: {fll['clip_loss']:.4f}, MSE Loss: {fll['mse_loss']:.4f}, CosSim Loss: {fll['cosine_similarity']:.4f}, Total: {fll['total']:.4f}"
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
            average_test_accuracy = (
                sum(
                    self.metrics["test"][t][-1]["accuracy"]
                    for t in self.metrics["test"]
                )
                / 3
            )
            average_final_layer_total_loss = (
                sum(
                    self.metrics["test"][t][-1]["final_layer_losses"]["total"]
                    for t in self.metrics["test"]
                )
                / 3
            )

            if (
                average_test_accuracy > self.highest_average_test_accuracy
                or average_final_layer_total_loss < self.lowest_final_layer_total_loss
            ):
                self.highest_average_test_accuracy = average_test_accuracy
                self.lowest_final_layer_total_loss = average_final_layer_total_loss
                self.highest_epoch = epoch
                self.highest_metrics = {
                    test: self.metrics["test"][test][-1]
                    for test in self.metrics["test"].keys()
                }
                self.log_print(
                    f"New highest test accuracy: {self.highest_average_test_accuracy:.4f}, "
                    f"lowest final-layer loss: {self.lowest_final_layer_total_loss:.4f}, "
                    f"epoch {self.highest_epoch}."
                )

            if epoch - self.highest_epoch > 10:
                self.log_print(
                    f"Early stopping at epoch {epoch}. Highest top10 at epoch {self.highest_epoch}."
                )
                break

        self.log_print("Training completed.")
        for test, mt in self.highest_metrics.items():
            self.log_print("\n")
            self.log_print(
                f"{test}: Acc: {mt['accuracy']:.4f}, Top5: {mt['top_5_accuracy']:.4f}, Top10: {mt['top_10_accuracy']:.4f}"
            )
            self.log_print(
                f"Loss: {mt['loss']:.4f}, Mel: {mt['mel_loss']:.4f}, "
                f"Clip: {mt['clip_loss']:.4f}, MSE: {mt['mse_loss']:.4f}, "
                f"CosSim: {mt['cosine_similarity_loss']:.4f}"
            )
            fll = mt["final_layer_losses"]
            self.log_print(
                f"FinLayer Clip: {fll['clip_loss']:.4f}, MSE: {fll['mse_loss']:.4f}, "
                f"CosSim: {fll['cosine_similarity']:.4f}, Total: {fll['total']:.4f}"
            )

            # Combine final-layer accuracies into a single string:
            self.log_print(
                f'FinLayer Acc: {mt["latent_alignment_metrics"]["clip_accuracy"][-1]:.4f}, '
                f'FinLayer Top5: {mt["latent_alignment_metrics"]["clip_top5_accuracy"][-1]:.4f}, '
                f'FinLayer Top10: {mt["latent_alignment_metrics"]["clip_top10_accuracy"][-1]:.4f}'
            )

    def run_batch(self, batch: AudioBatch, train: bool) -> tp.Tuple[dict, int]:
        """
        The main place we do forward/backward. We've added final-layer top-5 and top-10
        accuracy just like top-1.
        """
        brain_segments, audio_segments, recording = (
            batch.brain_segments["all"],
            batch.audio_segments,
            batch.recording,
        )
        brain_segments, audio_segments = self.discard_nan(
            brain_segments, audio_segments
        )

        # Accumulators for sums
        recording_loss = 0.0
        recording_mel_loss = 0.0
        recording_clip_loss = 0.0
        recording_cosine_similarity_loss = 0.0
        recording_mse_loss = 0.0
        recording_commitment_loss = 0.0

        # Per-layer alignment losses
        recording_latent_alignment_losses = {
            "cosine_similarity": [0.0 for _ in self.config.latent_alignment_layers],
            "mse_loss": [0.0 for _ in self.config.latent_alignment_layers],
            "clip_loss": [0.0 for _ in self.config.latent_alignment_layers],
            "total": [0.0 for _ in self.config.latent_alignment_layers],
        }

        # Layer-wise raw correct counts
        recording_latent_alignment_correct = [
            0.0 for _ in self.config.latent_alignment_layers
        ]
        recording_latent_alignment_top5 = [
            0.0 for _ in self.config.latent_alignment_layers
        ]
        recording_latent_alignment_top10 = [
            0.0 for _ in self.config.latent_alignment_layers
        ]

        # For mel-level clip correctness
        recording_correct = 0.0
        recording_top_5 = 0.0
        recording_top_10 = 0.0
        recording_perplexity = 0.0
        recording_temp = 0.0

        total_samples = brain_segments.shape[0]
        missed_recordings = 0
        missed_batches = 0

        conditions = {
            "study": f"{recording.study_name}",
            "subject": f"{recording.study_name}_{recording.subject_id}",
        }

        shuffle_idx = torch.randperm(total_samples)
        brain_segments = brain_segments[shuffle_idx]
        audio_segments = audio_segments[shuffle_idx]

        # We'll slice by batch_size
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

                    (
                        x,
                        quantizer_metrics,
                        channel_weights,
                        hidden_outputs,
                        encoder_hidden_states,
                    ) = self.model(
                        x=[brain_batch],
                        recording=[recording],
                        conditions=[conditions],
                        mel=[audio_batch],
                        train=train,
                        return_hidden_outputs=False,
                    )
                    del channel_weights, hidden_outputs, brain_batch

                    # Frozen
                    with torch.no_grad():
                        outputs = self.frozen_encoder(
                            nn.functional.pad(
                                audio_batch,
                                (0, 3000 - audio_batch.size(2)),
                                mode="constant",
                                value=0.0,
                            ),
                            output_hidden_states=(
                                self.config.latent_alignment_layers != [-1]
                            ),
                        )
                        if self.config.latent_alignment_layers == [-1]:
                            frozen_encoder_outputs = [
                                outputs.last_hidden_state[:, : audio_batch.size(2), :]
                            ]
                        else:
                            frozen_encoder_outputs = [
                                outputs.hidden_states[l][:, : audio_batch.size(2), :]
                                for l in self.model.layers_to_align
                            ]
                            del outputs
                            gc.collect()

                    # Mel alignment objectives
                    if self.config.mel_alignment_objectives["mse_loss"] > 0:
                        mse_l = self.mse_loss(pred=x, target=audio_batch)
                    else:
                        mse_l = torch.tensor(0.0).to(device)

                    if self.config.mel_alignment_objectives["cosine_similarity"] > 0:
                        cosim_l = self.cosine_similarity_loss(
                            x.transpose(1, 2), audio_batch.transpose(1, 2)
                        )
                    else:
                        cosim_l = torch.tensor(0.0).to(device)

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
                        self.config.mel_alignment_objectives["clip_loss"]
                        * mel_clip_loss
                        + self.config.mel_alignment_objectives["mse_loss"] * mse_l
                        + self.config.mel_alignment_objectives["cosine_similarity"]
                        * cosim_l
                    )

                    if quantizer_metrics is not None:
                        if "commitment_loss" in quantizer_metrics:
                            mel_loss += (
                                self.config.mel_alignment_objectives["commitment_loss"]
                                * quantizer_metrics["commitment_loss"]
                            )

                    # Latent alignment for each layer
                    la_losses = {
                        "cosine_similarity": [],
                        "mse_loss": [],
                        "clip_loss": [],
                        "total": [],
                    }
                    la_correct = []
                    la_top5 = []
                    la_top10 = []

                    for l_idx, (fro_out, hid_out) in enumerate(
                        zip(frozen_encoder_outputs, encoder_hidden_states)
                    ):
                        # cos
                        if (
                            self.config.latent_alignment_objectives["cosine_similarity"]
                            > 0
                        ):
                            la_cos = self.cosine_similarity_loss(fro_out, hid_out)
                        else:
                            la_cos = torch.tensor(0.0).to(device)

                        # mse
                        if self.config.latent_alignment_objectives["mse_loss"] > 0:
                            la_mse = self.mse_loss(pred=hid_out, target=fro_out)
                        else:
                            la_mse = torch.tensor(0.0).to(device)

                        # clip
                        if self.config.latent_alignment_objectives["clip_loss"] > 0:
                            la_clip_results = self.clip_loss_latent(
                                hid_out.transpose(1, 2), fro_out.transpose(1, 2)
                            )
                            la_clip_loss, la_clip_metrics = (
                                la_clip_results["loss"],
                                la_clip_results["metrics"],
                            )
                        else:
                            la_clip_loss = torch.tensor(0.0).to(device)
                            la_clip_metrics = {
                                "correct": 0,
                                "top_5_correct": 0,
                                "top_10_correct": 0,
                            }

                        la_tot = (
                            self.config.latent_alignment_objectives["cosine_similarity"]
                            * la_cos
                            + self.config.latent_alignment_objectives["mse_loss"]
                            * la_mse
                            + self.config.latent_alignment_objectives["clip_loss"]
                            * la_clip_loss
                        )
                        la_losses["cosine_similarity"].append(la_cos)
                        la_losses["mse_loss"].append(la_mse)
                        la_losses["clip_loss"].append(la_clip_loss)
                        la_losses["total"].append(la_tot)

                        la_correct.append(la_clip_metrics["correct"])
                        la_top5.append(la_clip_metrics["top_5_correct"])
                        la_top10.append(la_clip_metrics["top_10_correct"])

                    total_loss = sum(la_losses["total"]) + mel_loss

                    if not torch.isnan(total_loss).any():
                        if train:
                            self.scaler.scale(total_loss).backward()

                            # Clip gradients
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), max_norm=5.0
                            )

                            if self.adalora_steps >= self.config.adalora_config.tinit:
                                self.model.encoder.base_model.update_and_allocate(
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

                        # Accumulate
                        recording_loss += total_loss.detach().cpu().item()
                        recording_mse_loss += mse_l.detach().cpu().item()
                        recording_cosine_similarity_loss += (
                            cosim_l.detach().cpu().item()
                        )
                        recording_clip_loss += mel_clip_loss.detach().cpu().item()
                        recording_mel_loss += mel_loss.detach().cpu().item()

                        if quantizer_metrics is not None:
                            if "perplexity" in quantizer_metrics:
                                perplex = (
                                    quantizer_metrics["perplexity"]
                                    .detach()
                                    .cpu()
                                    .mean()
                                )
                                recording_perplexity += perplex.item()
                            if "temp" in quantizer_metrics:
                                recording_temp += (
                                    quantizer_metrics["temp"].detach().cpu().item()
                                )
                            if "commitment_loss" in quantizer_metrics:
                                recording_commitment_loss += (
                                    quantizer_metrics["commitment_loss"]
                                    .detach()
                                    .cpu()
                                    .item()
                                )

                        # sum up layerwise
                        for key in la_losses:
                            for i_l, val in enumerate(la_losses[key]):
                                recording_latent_alignment_losses[key][i_l] += (
                                    val.detach().cpu().item()
                                )

                        # sum up raw correct counts
                        for i_l in range(len(self.config.latent_alignment_layers)):
                            recording_latent_alignment_correct[i_l] += la_correct[i_l]
                            recording_latent_alignment_top5[i_l] += la_top5[i_l]
                            recording_latent_alignment_top10[i_l] += la_top10[i_l]

                        # mel-level clip metrics
                        recording_correct += mel_clip_metrics["correct"]
                        recording_top_5 += mel_clip_metrics["top_5_correct"]
                        recording_top_10 += mel_clip_metrics["top_10_correct"]

                    else:
                        self.log_print(
                            f"NaN loss on {recording.study_name} subj {recording.subject_id} task {recording.task_id}."
                        )
                        missed_recordings += end - start
                        missed_batches += 1

                except Exception as ex:
                    self.log_print(f"Error in run_batch: {ex}")
                    missed_recordings += end - start
                    missed_batches += 1
                    raise ex

        if train:
            self.scheduler.step()

        total_samples -= missed_recordings
        batches = len(batch_indices) - missed_batches

        # Average the losses by #batches
        for key in recording_latent_alignment_losses:
            for i_l in range(len(recording_latent_alignment_losses[key])):
                recording_latent_alignment_losses[key][i_l] /= batches

        # Convert raw correct to accuracy by dividing by total_samples
        latent_alignment_metrics = {
            "clip_accuracy": [],
            "clip_top5_accuracy": [],
            "clip_top10_accuracy": [],
        }
        for i_l in range(len(self.config.latent_alignment_layers)):
            if total_samples > 0:
                clip_acc = recording_latent_alignment_correct[i_l] / total_samples
                clip_top5_acc = recording_latent_alignment_top5[i_l] / total_samples
                clip_top10_acc = recording_latent_alignment_top10[i_l] / total_samples
            else:
                clip_acc, clip_top5_acc, clip_top10_acc = 0.0, 0.0, 0.0

            latent_alignment_metrics["clip_accuracy"].append(clip_acc)
            latent_alignment_metrics["clip_top5_accuracy"].append(clip_top5_acc)
            latent_alignment_metrics["clip_top10_accuracy"].append(clip_top10_acc)

        final_layer_losses = {
            key: recording_latent_alignment_losses[key][-1]
            for key in recording_latent_alignment_losses
        }

        metrics = {
            "loss": recording_loss / batches,
            "mel_loss": recording_mel_loss / batches,
            "clip_loss": recording_clip_loss / batches,
            "cosine_similarity_loss": recording_cosine_similarity_loss / batches,
            "mse_loss": recording_mse_loss / batches,
            "commitment_loss": recording_commitment_loss / batches,
            "perplexity": recording_perplexity / batches,
            "alignment_losses": recording_latent_alignment_losses,
            "final_layer_losses": final_layer_losses,
            "latent_alignment_metrics": latent_alignment_metrics,
            "accuracy": (recording_correct / total_samples) if total_samples else 0.0,
            "top_5_accuracy": (
                (recording_top_5 / total_samples) if total_samples else 0.0
            ),
            "top_10_accuracy": (
                (recording_top_10 / total_samples) if total_samples else 0.0
            ),
        }

        self.logger.info(
            f"{recording.study_name} {recording.subject_id} sess {recording.session_id}, Loss: {metrics['loss']:.4f}"
        )
        self.logger.info(
            f"Mel Loss: {metrics['mel_loss']:.4f}, Clip Loss: {metrics['clip_loss']:.4f}, "
            f"MSE: {metrics['mse_loss']:.4f}, Commit: {metrics['commitment_loss']:.4f}, "
            f"CosSim: {metrics['cosine_similarity_loss']:.4f}"
        )
        self.logger.info(
            f"Perplex: {metrics['perplexity']:.4f}, Acc: {metrics['accuracy']:.4f}, "
            f"Top5: {metrics['top_5_accuracy']:.4f}, Top10: {metrics['top_10_accuracy']:.4f}"
        )

        # Print final-layer alignment losses:
        fll = final_layer_losses
        self.logger.info(
            f"FinalLayer Clip Loss: {fll['clip_loss']:.4f}, MSE: {fll['mse_loss']:.4f}, "
            f"CosSim: {fll['cosine_similarity']:.4f}, Tot: {fll['total']:.4f}"
        )
        # Print final-layer accuracy, top5, top10
        if "clip_accuracy" in latent_alignment_metrics:
            last_l = len(latent_alignment_metrics["clip_accuracy"]) - 1
            final_acc = latent_alignment_metrics["clip_accuracy"][last_l]
            final_acc_5 = latent_alignment_metrics["clip_top5_accuracy"][last_l]
            final_acc_10 = latent_alignment_metrics["clip_top10_accuracy"][last_l]
            self.logger.info(
                f"LatentAlign final-layer clip acc: {final_acc:.4f}, "
                f"top5: {final_acc_5:.4f}, top10: {final_acc_10:.4f}"
            )

        return metrics, total_samples

    def test(self, buffer_size: int, num_workers: int, max_cache_size: int):
        self.model.eval().to(self.device)
        self.set_seed(int(self.config.seed))
        start_time = time.time()

        test_datasets, test_sizes, test_dataloader = {}, {}, {}
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
            )
            test_dataloader[test].start_fetching(test_datasets[test], cache=True)

        for test in test_datasets:
            all_metrics, total_batches = [], 0
            while True:
                batch = test_dataloader[test].get_recording()
                if batch is None:
                    break
                try:
                    results, num_b = self.run_batch(batch, train=False)
                    all_metrics.append(results)
                    total_batches += num_b
                except Exception as e:
                    self.log_print(f"Error in testing {test}, skipping. {e}")
                    test_sizes[test] -= 1
                    continue
                del batch
                gc.collect()

            if test_sizes[test] == 0:
                continue

            fm = {
                "loss": sum(m["loss"] for m in all_metrics) / test_sizes[test],
                "mel_loss": sum(m["mel_loss"] for m in all_metrics) / test_sizes[test],
                "clip_loss": sum(m["clip_loss"] for m in all_metrics)
                / test_sizes[test],
                "mse_loss": sum(m["mse_loss"] for m in all_metrics) / test_sizes[test],
                "cosine_similarity_loss": sum(
                    m["cosine_similarity_loss"] for m in all_metrics
                )
                / test_sizes[test],
                "commitment_loss": sum(m["commitment_loss"] for m in all_metrics)
                / test_sizes[test],
                "perplexity": sum(m["perplexity"] for m in all_metrics)
                / test_sizes[test],
                "alignment_losses": {
                    key: [
                        sum(m["alignment_losses"][key][i] for m in all_metrics)
                        / test_sizes[test]
                        for i in range(len(self.config.latent_alignment_layers))
                    ]
                    for key in all_metrics[0]["alignment_losses"]
                },
                "final_layer_losses": {
                    key: sum(m["final_layer_losses"][key] for m in all_metrics)
                    / test_sizes[test]
                    for key in all_metrics[0]["final_layer_losses"]
                },
                "latent_alignment_metrics": {},
                "accuracy": sum(m["accuracy"] for m in all_metrics) / test_sizes[test],
                "top_5_accuracy": sum(m["top_5_accuracy"] for m in all_metrics)
                / test_sizes[test],
                "top_10_accuracy": sum(m["top_10_accuracy"] for m in all_metrics)
                / test_sizes[test],
            }

            # If we have alignment metrics
            if "latent_alignment_metrics" in all_metrics[0]:
                lam_keys = all_metrics[0]["latent_alignment_metrics"].keys()
                final_lam = {}
                for lam_key in lam_keys:
                    layer_vals = []
                    for idx in range(len(self.config.latent_alignment_layers)):
                        total_val = 0.0
                        for mm in all_metrics:
                            total_val += mm["latent_alignment_metrics"][lam_key][idx]
                        layer_vals.append(total_val / test_sizes[test])
                    final_lam[lam_key] = layer_vals
                fm["latent_alignment_metrics"] = final_lam

            self.metrics["test"][test].append(fm)

            self.log_no_print(
                f"Test {test} done. Loss: {fm['loss']:.4f}, Mel: {fm['mel_loss']:.4f}"
            )
            self.log_no_print(
                f"Clip: {fm['clip_loss']:.4f}, MSE: {fm['mse_loss']:.4f}, "
                f"CosSim: {fm['cosine_similarity_loss']:.4f}, Commit: {fm['commitment_loss']:.4f}"
            )
            self.log_no_print(
                f"Perplex: {fm['perplexity']:.4f}, Acc: {fm['accuracy']:.4f}, "
                f"Top5: {fm['top_5_accuracy']:.4f}, Top10: {fm['top_10_accuracy']:.4f}"
            )
            fll = fm["final_layer_losses"]
            self.log_no_print(
                f"FinLayer Clip: {fll['clip_loss']:.4f}, MSE: {fll['mse_loss']:.4f}, "
                f"CosSim: {fll['cosine_similarity']:.4f}, Tot: {fll['total']:.4f}"
            )

            # Also print final-layer clip acc, top5, top10
            if "latent_alignment_metrics" in fm:
                lam = fm["latent_alignment_metrics"]
                if "clip_accuracy" in lam:
                    last_l = len(lam["clip_accuracy"]) - 1
                    final_acc = lam["clip_accuracy"][last_l]
                    final_acc_5 = lam["clip_top5_accuracy"][last_l]
                    final_acc_10 = lam["clip_top10_accuracy"][last_l]
                    self.log_no_print(
                        f"Final layer latent clip acc: {final_acc:.4f}, "
                        f"top5: {final_acc_5:.4f}, top10: {final_acc_10:.4f}"
                    )

            test_dataloader[test].stop()

        elaps = (time.time() - start_time) / 60
        self.log_print(f"Testing done in {elaps:.2f}m.")

    def get_dataloader(self, buffer_size, num_workers, max_cache_size):
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
            batch_types={"audio": num_workers},
            batch_kwargs={
                "audio": {
                    "max_random_shift": self.config.max_random_shift,
                    "window_size": self.config.window_size,
                    "window_stride": self.config.window_stride,
                    "audio_sample_rate": self.config.audio_sample_rate,
                    "hop_length": self.config.hop_length,
                    "audio_processor": self.config.audio_model,
                }
            },
        )
        return dataloader

    def discard_nan(
        self,
        brain: torch.Tensor,
        audio: torch.Tensor,
    ):
        valid_mask = ~(
            torch.isnan(brain).any(dim=(1, 2)) | torch.isnan(audio).any(dim=(1, 2))
        )
        if valid_mask.all():
            return brain, audio
        filtered_brain = brain[valid_mask]
        filtered_audio = audio[valid_mask]
        if filtered_brain.shape[0] != filtered_audio.shape[0]:
            raise ValueError("Filtered brain and audio must match in length")
        return filtered_brain, filtered_audio

    def pre_process_all_recordings(
        self, buffer_size: int, num_workers: int, max_cache_size: int
    ):
        if self.recordings is None:
            self.partition_datasets()
        dataloader = self.get_dataloader(buffer_size, num_workers, max_cache_size)
        total_rec = len(self.recordings)
        pbar = tqdm(total=total_rec, desc="Loading recordings")

        dataloader.start_fetching(self.recordings)
        done = 0
        while True:
            r = dataloader.get_recording()
            if r is None:
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
            self.model.encoder.save_pretrained(f"{ckp_path}/adalora_adapter")

            # Save metrics
            torch.save(
                {
                    "metrics": self.metrics,
                    "error": str(self.error) if self.error else "No errors.",
                    "highest_epoch": self.highest_epoch,
                    "highest_metrics": self.highest_metrics,
                    "lowest_final_layer_total_loss": self.lowest_final_layer_total_loss,
                    "highest_average_test_accuracy": self.highest_average_test_accuracy,
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
    config = TrainingConfigV1(
        brain_encoder_config=None,
        data_partition=None,
    ).from_dict(config_dict)

    ts = TrainingSessionV1(
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

    whisper_model = WhisperModel.from_pretrained(
        config.audio_model, low_cpu_mem_usage=True, use_safetensors=True
    )
    enc = whisper_model.get_encoder()
    enc._freeze_parameters()
    del whisper_model.decoder
    del whisper_model

    peft_enc = PeftModel.from_pretrained(enc, adalora_adapter_path)
    adalora_config = PeftConfig.from_pretrained(adalora_adapter_path)
    ts.model.adalora_config = adalora_config
    ts.model.encoder = peft_enc

    metrics_path = os.path.join(save_path, "metrics.pt")
    if os.path.exists(metrics_path):
        loaded = torch.load(metrics_path, map_location="cpu")
        if "metrics" in loaded:
            ts.metrics = loaded["metrics"]
        else:
            ts.metrics = {}
        ts.highest_epoch = loaded.get("highest_epoch", 0)
        ts.highest_metrics = loaded.get("highest_metrics", {})
        ts.lowest_final_layer_total_loss = loaded.get(
            "lowest_final_layer_total_loss", float("inf")
        )
        ts.highest_average_test_accuracy = loaded.get(
            "highest_average_test_accuracy", 0
        )
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
