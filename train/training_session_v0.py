import gc
import random
import time
from tkinter import E
from numpy import int0
from tqdm import tqdm
import typing as tp
import json
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
import os
import torch

from dataloader import DataLoader
from dataloader.audio_batch import AudioBatch
from losses.clip import CLIPLoss
from losses.mse import mse_loss_per_batch
from config import TrainingConfigV0
from train.training_session import TrainingSession
from models.simpleconv import SimpleConv

device = "cuda"


class TrainingSessionV0(TrainingSession):
    def __init__(
        self,
        config: TrainingConfigV0 = None,
        studies: tp.Dict[str, str] = None,
        data_path: str = "/home/ubuntu/brain-decoding/data",
        save_path: str = "/home/ubuntu/brain-decoding/saves",
        clear_cache: bool = False,
        cache_enabled: bool = True,
        max_cache_size: int = 100,
    ):
        """Initializes a training session with the provided configuration and data.
        This version deals with audio batches.

        Arguments:
            config -- The configuration for the training session.
            studies -- dict of studies, batch type. Partition policy determined in TrainingConfig
                    Batch type determines how to load data from study.

            data_path -- The path to the data directory.
            save_path -- The path to the directory where the model and logs will be saved.
            clear_cache -- Whether to clear the cache for the studies.
            cache_enabled -- Whether to enable caching for the studies.
            max_cache_size -- The maximum number of stimulis in cache.
        """

        super().__init__(
            config=config,
            studies=studies,
            data_path=data_path,
            save_path=save_path,
            clear_cache=clear_cache,
            cache_enabled=cache_enabled,
            max_cache_size=max_cache_size,
        )

        # Set conditions
        if self.config.brain_encoder_config.conditions is not None:
            if "study" in self.config.brain_encoder_config.conditions:
                self.config.brain_encoder_config.conditions["study"] = list(
                    studies.keys()
                )

            if "subjects" in self.config.brain_encoder_config.conditions:
                subjects = set()
                for recording in self.recordings:
                    subjects.add(f"{recording.study_name}_{recording.subject_id}")
                self.config.brain_encoder_config.conditions["subjects"] = list(subjects)

        self.model = SimpleConv(self.config.brain_encoder_config)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scaler = torch.amp.GradScaler(device=device)
        self.clip_loss, self.mse_loss = CLIPLoss(), mse_loss_per_batch

    def train(
        self,
        device: str,
        buffer_size: int,
        num_workers: int,
        max_cache_size: int,
        current_epoch: int = 0,
    ):
        """Max cache size for the cache dir in GB"""

        # Set all training parameters
        self.device = device
        gpu_ok = False
        torch.set_float32_matmul_precision("high")
        training_size = len(self.dataset["train"])
        self.model.to(device)
        self.clip_loss.to(device)

        # Check if GPU is NVIDIA V100, A100, or H100
        if torch.cuda.is_available():
            device_cap = torch.cuda.get_device_capability()
            if device_cap in ((7, 0), (8, 0), (9, 0)):
                gpu_ok = True
        if not gpu_ok:
            self.log_print(
                "GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower than expected."
            )

        # Fetch recordings
        dataloader = self.get_dataloader(
            buffer_size=buffer_size,
            num_workers=num_workers,
            max_cache_size=max_cache_size,
        )

        for epoch in range(current_epoch + 1, self.config.epochs + 1):
            try:
                self.model.to(device).train()
                epoch_start_time = time.time()

                # Shuffle for each epoch, and start fetching
                epoch_training_dataset, remaining = (
                    self.dataset["train"].copy(),
                    training_size,
                )
                # For reproducibility
                self.set_seed(int(self.config.seed + epoch))
                random.shuffle(epoch_training_dataset)
                dataloader.start_fetching(epoch_training_dataset, cache=True)

            except Exception as e:
                self.log_print(f"Error in epoch {epoch} during initialization, {e}")
                self.save(f"error_epoch_{epoch}")

            pbar = tqdm(total=training_size, desc="Training Epoch " + str(epoch))
            
            # Run each batch
            while True:

                batch = dataloader.get_recording()
                if batch is None:
                    break

                try:
                    start_time = time.time()
                    results = self.run_batch(batch, train=True)
                    self.metrics["train"].append(results)

                    # Don't print, just log
                    self.logger.info(
                        f"Epoch {epoch}, Remaining {remaining}/{training_size}. Runtime {time.time() - start_time:.2f}s."
                    )
                    self.logger.info(
                        f'Loss: {results["loss"]:.4f}, Clip Loss: {results["clip_loss"]:.4f}, MSE Loss: {results["mse_loss"]:.4f}, Commitment Loss: {results["commitment_loss"]:.4f}'
                    )
                    self.logger.info(
                        f'Accuracy: {results["accuracy"]:.4f}, Top 1: {results["top_1_accuracy"]:.4f}, Top 5: {results["top_5_accuracy"]:.4f}, Top 10: {results["top_10_accuracy"]:.4f}, Perplexity: {results["perplexity"]:.4f}'
                    )
                    remaining -= 1
                except Exception as e:
                    # Do log errors
                    self.log_print(
                        f"Error in epoch {epoch}, {batch.recording.study_name} {batch.recording.subject_id} {batch.recording.session_id} {batch.recording.task_id}. Skipping. {e}"
                    )
                    raise e
                    continue
                
                pbar.update(1)

            elapsed_minutes = (time.time() - epoch_start_time) / 60
            self.log_print(
                f"Epoch {epoch} completed in {elapsed_minutes:.2f}m. {elapsed_minutes / training_size:.2f}m per recording."
            )

            # Testing
            try:
                self.log_print(f"Testing at epoch {epoch}")
                with torch.no_grad():
                    self.test(
                        buffer_size=buffer_size,
                        num_workers=num_workers,
                        max_cache_size=max_cache_size,
                    )
            except Exception as e:
                self.log_print(f"Error in epoch {epoch} during testing, {e}")
                self.save(f"error_epoch_{epoch}")
                raise e

            # Save model
            self.save(f"epoch_{epoch}")

        self.log_print("Training completed.")

    def run_batch(self, batch: AudioBatch, train: bool) -> tp.Dict[str, float]:
        """
        Per recording processing for training and testing. Returns average metrics
        and losses for the recording. Returns metrics on CPU. 
        """

        # Some processing to ensure dims match
        brain_segments, audio_segments, recording = batch.brain_segments, batch.audio_segments, batch.recording
        brain_segments, audio_segments = self.discard_nan(
            brain_segments["all"], audio_segments
        )

        # Initialize recording metrics
        (
            recording_loss,
            recording_clip_loss,
            recording_mse_loss,
            recording_commitment_loss,
        ) = (0, 0, 0, 0)

        (total, missed_recordings, missed_batches) = (
            brain_segments.shape[0],
            0,
            0,
        )
        (
            recording_correct,
            recording_top_1,
            recording_top_5,
            recording_top_10,
            recording_perplexity,
            recording_temp,
        ) = (
            0,
            0,
            0,
            0,
            0,
            0,
        )

        # Models config decides if it is used
        conditions = {
            "study": str(recording.study_name),
            "subject": str(recording.study_name) + "_" + str(recording.subject_id),
        }

        # Shuffle segments
        shuffle_indices = torch.randperm(brain_segments.shape[0])
        brain_segments, audio_segments = (
            brain_segments[shuffle_indices].to(self.device),
            audio_segments[shuffle_indices].to(self.device),
        )  # [B, C, T], [B, mel_bins, T]

        # Process by specified batch size
        batch_indices = [
            (i, min(i + self.config.batch_size, total))
            for i in range(0, total, self.config.batch_size)
        ]

        with torch.amp.autocast(dtype=torch.bfloat16, device_type=device):

            for start, end in batch_indices:

                try:

                    if train:
                        self.optimizer.zero_grad()

                    # Slice by batch
                    brain_batch, audio_batch = (
                        brain_segments[start:end],
                        audio_segments[start:end],
                    )

                    # Forward pass
                    (output, quantizer_metrics) = self.model(
                        x=brain_batch,
                        recording=recording,
                        conditions=conditions,
                        mel=audio_batch,
                        train=True,
                    )  # [B, C, T]

                    # Compute loss
                    mse_loss = self.mse_loss(pred=output, target=audio_batch)
                    
                    clip_results = self.clip_loss(x_1=output, x_2=audio_batch)
                    clip_loss, clip_metrics = (
                        clip_results["loss"],
                        clip_results["metrics"],
                    )

                    # Sum loss based on config
                    if self.config.use_clip_loss and self.config.use_mse_loss:
                        loss = ((1 - self.config.alpha) * mse_loss) + (
                            self.config.alpha * clip_loss
                        )
                    elif not self.config.use_clip_loss and self.config.use_mse_loss:
                        loss = mse_loss
                    elif self.config.use_clip_loss and not self.config.use_mse_loss:
                        loss = clip_loss

                    if quantizer_metrics is not None:
                        if "commitment_loss" in quantizer_metrics:
                            loss += quantizer_metrics["commitment_loss"]

                    # Backward pass
                    if not torch.isnan(loss).any():

                        if train:
                            self.scaler.scale(loss).backward()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()

                        # Store losses, move to CPU
                        recording_loss += loss.detach().to("cpu").item()
                        recording_clip_loss += clip_loss.detach().to("cpu").item()
                        recording_mse_loss += mse_loss.detach().to("cpu").item()

                        # Store metrics, already on CPU
                        recording_correct += clip_metrics["correct"]
                        recording_top_1 += clip_metrics["top_1_correct"]
                        recording_top_5 += clip_metrics["top_5_correct"]
                        recording_top_10 += clip_metrics["top_10_correct"]

                        # Quantizer metrics
                        if quantizer_metrics is not None:
                            if "perplexity" in quantizer_metrics:
                                perplexity = (
                                    quantizer_metrics["perplexity"]
                                    .detach()
                                    .to("cpu")
                                    .mean(dim=0)
                                )
                                recording_perplexity += perplexity.item()
                            if "temp" in quantizer_metrics:
                                recording_temp += (
                                    quantizer_metrics["temp"].detach().to("cpu").item()
                                )
                            if "commitment_loss" in quantizer_metrics:
                                recording_commitment_loss += (
                                    quantizer_metrics["commitment_loss"]
                                    .detach()
                                    .to("cpu")
                                    .item()
                                )
                    else:
                        self.logger.info(
                            f"Loss is NaN for {recording.study_name} {recording.subject_id} {recording.session_id} {recording.task_id}."
                        )
                        missed_recordings += end - start
                        missed_batches += 1

                except Exception as e:
                    self.logger.info(
                        f"Error in processing {recording.study_name} {recording.subject_id} {recording.session_id} {recording.task_id}."
                    )
                    missed_recordings += end - start
                    missed_batches += 1
                    raise e
                    continue

        gc.collect()
        torch.cuda.empty_cache()

        # Correct for missed recordings and batches
        total -= missed_recordings
        batches = len(batch_indices) - missed_batches

        # Loss divided by batches, metrics by total
        metrics = {
            "loss": recording_loss,
            "clip_loss": recording_clip_loss,
            "mse_loss": recording_mse_loss,
            "commitment_loss": (
                recording_commitment_loss
            ),
            "perplexity": recording_perplexity,
            "accuracy": recording_correct,
            "top_1_accuracy": recording_top_1,
            "top_5_accuracy": recording_top_5,
            "top_10_accuracy": recording_top_10,
        }
        
        for k, v in metrics.items():
            metrics[k] = v / batches if batches > 0 else 0
            
        return metrics

    def test(self, buffer_size: int, num_workers: int, max_cache_size: int):
        """Max cache size in GB"""

        self.model.eval().to(self.device)
        self.set_seed(int(self.config.seed))
        test_start_time = time.time()

        test_datasets, test_dataloader, test_sizes = {}, {}, {}

        # Create dataset and loader
        for test in self.dataset["test"].keys():
            # Randomly subsample recordings for each type of test
            if len(self.dataset["test"][test]) < self.config.random_test_size:
                test_datasets[test] = self.dataset["test"][test]
            else:
                test_datasets[test] = random.sample(
                    self.dataset["test"][test], self.config.random_test_size
                )
                
            test_sizes[test] = len(test_datasets[test])

            test_dataloader[test] = self.get_dataloader(
                buffer_size=len(test_sizes[test]),
                num_workers=num_workers,
                max_cache_size=max_cache_size,
            )

        
        # Run tests
        for test in test_datasets.keys():
            
            test_dataloader[test].start_fetching(test_datasets[test], cache=True)

            i = 0
            while True:

                batch = test_dataloader[test].get_recording()
                if batch is None:
                    break

                try:

                    start_time = time.time()

                    results = self.run_batch(batch, train=False)
                    self.metrics["test"][test].append(results)

                    # Log results
                    self.logger.info(
                        f"Testing {test} {i}/{test_sizes[test]}. Runtime {time.time() - start_time:.2f}s."
                    )
                    self.logger.info(
                        f'Loss: {results["loss"]:.4f}, Clip Loss: {results["clip_loss"]:.4f}, MSE Loss: {results["mse_loss"]:.4f}, Commitment Loss: {results["commitment_loss"]:.4f}'
                    )
                    self.logger.info(
                        f'Accuracy: {results["accuracy"]:.4f}, Top 1: {results["top_1_accuracy"]:.4f}, Top 5: {results["top_5_accuracy"]:.4f}, Top 10: {results["top_10_accuracy"]:.4f}, Perplexity: {results["perplexity"]:.4f}'
                    )
                    i += 1
                    
                except Exception as e:
                    self.log_print(
                        f"Error in testing {test}, {batch.recording.study_name} {batch.recording.subject_id} {batch.recording.session_id} {batch.recording.task_id}. Skipping."
                    )
                    test_sizes[test] -= 1
                    continue

        # Log info
        elapsed_minutes = (time.time() - test_start_time) / 60
        self.logger.info(f"Testing completed in {elapsed_minutes:.2f}m.")
        return

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
        """
        If any nan in brain or audio data, discard the batch.

        Arguments:
            brain -- The brain data tensor, [B, C, T]
            audio -- The audio data, [B, mel_bins, T]
        """

        valid_mask = ~(
            torch.isnan(brain).any(dim=(1, 2)) | torch.isnan(audio).any(dim=(1, 2))
        )

        if valid_mask.all():
            return brain, audio

        # Apply the same mask to both tensors
        filtered_brain = brain[valid_mask]
        filtered_audio = audio[valid_mask]

        if filtered_brain.shape[0] != filtered_audio.shape[0]:
            raise ValueError(
                "Filtered brain and audio data must have the same number of samples"
            )

        return filtered_brain, filtered_audio

    def pre_process_all_recordings(
        self, buffer_size: int, num_workers: int, max_cache_size: int
    ):
        """Pre-processes all data and saves as .pt in cache at once."""

        if self.recordings is None:
            self.partition_datasets()

        dataloader = self.get_dataloader(
            buffer_size, num_workers, max_cache_size
        )

        total_recordings, remaining = len(self.recordings), len(self.recordings)
        pbar = tqdm(total=total_recordings, desc="Loading recordings")

        dataloader.start_fetching(self.recordings)

        while True:
            recording = dataloader.get_recording()
            if recording is None:
                break
            remaining -= 1
            pbar.update(1)

    def save(self, name: str):
        """Saves the model and logs to the save path."""
        with torch.no_grad():

            # Training session config
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
                
            
            config = self.config.to_dict()
            with open(self.save_path + "/training_config.json", "w") as json_file:
                json.dump(config, json_file, indent=4)
            checkpoint_path = f"{self.save_path}/{name}"
            os.makedirs(checkpoint_path, exist_ok=True)

            # Save model
            torch.save(
                {
                    "config": self.config.to_dict(),
                    "model": self.model.cpu().state_dict(),
                    "conditions": self.model.condition_to_idx,
                    "optimizer": self.optimizer.state_dict(),
                    "scaler": self.scaler.state_dict(),
                    "error": str(self.error) if self.error else "No errors.",
                },
                f"{checkpoint_path}/model.pt",
            )

            # Save metrics
            torch.save(
                {
                    "metrics": self.metrics,
                    "error": str(self.error) if self.error else "No errors.",
                },
                f"{checkpoint_path}/metrics.pt",
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
    cache_enabled: bool = True,
    max_cache_size: int = 100,
):
    """Loads a training session from the save path."""
    # Load training session config

    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Save path {save_path} does not exist.")

    try:
        load = torch.load(f"{save_path}/model.pt")
        config = load["config"]
        config = TrainingConfigV0(brain_encoder_config=None, data_partition=None).from_dict(config)

        training_session = TrainingSessionV0(
            config=config,
            studies=studies,
            data_path=data_path,
            save_path="temp",
            clear_cache=clear_cache,
            cache_enabled=cache_enabled,
            max_cache_size=max_cache_size,
        )
        training_session.save_path = save_path

        # Load model
        training_session.model.load_state_dict(load["model"])
        training_session.optimizer.load_state_dict(load["optimizer"])
        training_session.scaler.load_state_dict(load["scaler"])
        training_session.error = load["error"]

        # Load metrics
        metrics_path = os.path.join(save_path, "metrics.pt")
        if os.path.exists(metrics_path):
            metrics = torch.load(metrics_path)
            training_session.metrics = metrics.get("metrics", {})
        else:
            training_session.metrics = {}
            training_session.logger.warning(
                f"Metrics file not found at {metrics_path}."
            )

        if training_session.model.condition_to_idx != load["conditions"]:
            raise ValueError("Condition to idx mismatch.")

        return training_session

    except Exception as e:
        raise ValueError(f"Error loading training session config, {e}")
