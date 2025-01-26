import gc
import random
import time
from tqdm import tqdm
import typing as tp
import json
from torch.optim import AdamW
import os
import torch

from dataloader import DataLoader
from dataloader.audio_batch import AudioBatch
from losses.mse import mse_loss_per_batch
from config import TrainingConfigV0
from train.training_session import TrainingSession
from models.simpleconv import SimpleConv
from models.spectralconv import SpectralConv
from config import SimpleConvConfig, SpectralConvConfig

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
        cache_name: str = "cache",
    ):
        """
        Initializes a training session with the provided configuration and data.
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
            cache_name=cache_name,
        )

        if isinstance(self.config.brain_encoder_config, SimpleConvConfig):
            self.model = SimpleConv(self.config.brain_encoder_config)
        elif isinstance(self.config.brain_encoder_config, SpectralConvConfig):
            self.model = SpectralConv(self.config.brain_encoder_config)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scaler = torch.amp.GradScaler(device=device)
        self.clip_loss, self.mse_loss = self.model.clip_loss, mse_loss_per_batch

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
        self.model.to(device)
        self.clip_loss.to(device)
        training_size = len(self.dataset["train"])

        for epoch in range(current_epoch + 1, self.config.epochs + 1):
            try:
                self.model.to(device).train()
                epoch_start_time = time.time()

                # Shuffle for each epoch, and start fetching
                epoch_training_dataset, remaining = (
                    self.dataset["train"].copy(),
                    training_size,
                )
                # Fetch recordings
                dataloader = self.get_dataloader(
                    buffer_size=buffer_size,
                    num_workers=num_workers,
                    max_cache_size=max_cache_size,
                )
                # For reproducibility
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

            # Run each batch
            while True:

                batch = dataloader.get_recording()
                if batch is None:
                    break

                try:
                    results, num_batches = self.run_batch(batch, train=True)
                    all_metrics.append(results)
                    total_batches += num_batches

                except Exception as e:
                    # Do log errors
                    self.log_print(
                        f"Error in epoch {epoch}, {batch.recording.study_name} {batch.recording.subject_id} {batch.recording.session_id} {batch.recording.task_id}. Skipping. {e}"
                    )
                    self.save(f"error_epoch_{epoch}")
                    raise e

                del batch
                gc.collect()

                pbar.update(1)
            pbar.close()

            # it is not "accuracy" but 'correct' until averaged
            final_metrics = {
                "loss": sum([batch["loss"] for batch in all_metrics]) / training_size,
                "clip_loss": sum([batch["clip_loss"] for batch in all_metrics])
                / training_size,
                "mse_loss": sum([batch["mse_loss"] for batch in all_metrics])
                / training_size,
                "commitment_loss": sum(
                    [batch["commitment_loss"] for batch in all_metrics]
                )
                / training_size,
                "accuracy": sum([batch["correct"] for batch in all_metrics])
                / total_batches,
                "top_5_accuracy": sum([batch["top_5_correct"] for batch in all_metrics])
                / total_batches,
                "top_10_accuracy": sum(
                    [batch["top_10_correct"] for batch in all_metrics]
                )
                / total_batches,
            }
            self.metrics["train"].append(final_metrics)

            self.log_print(
                f'Epoch {epoch} completed. Loss: {final_metrics["loss"]:.4f}, Clip Loss: {final_metrics["clip_loss"]:.4f}, MSE Loss: {final_metrics["mse_loss"]:.4f}'
            )
            self.log_print(
                f'Accuracy: {final_metrics["accuracy"]:.4f}, Top 5: {final_metrics["top_5_accuracy"]:.4f}, Top 10: {final_metrics["top_10_accuracy"]:.4f}'
            )

            # Testing
            try:
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

            elapsed_minutes = (time.time() - epoch_start_time) / 60
            self.log_print(
                f"Epoch {epoch} completed in {elapsed_minutes:.2f}m. {elapsed_minutes / training_size:.2f}m per recording."
            )

            # Save model
            self.save(f"epoch_{epoch}")

            # Early stopping
            average_test_accuracy = (
                sum(
                    [
                        self.metrics["test"][test][-1]["accuracy"]
                        for test in self.metrics["test"].keys()
                    ]
                )
                / 3
            )

            if average_test_accuracy > self.highest_average_test_accuracy:

                self.highest_average_test_accuracy = average_test_accuracy
                self.highest_epoch = epoch

                self.highest_metrics = {
                    test: self.metrics["test"][test][-1]
                    for test in self.metrics["test"].keys()
                }

            if epoch - self.highest_epoch > 10:
                self.log_print(
                    f"Early stopping at epoch {epoch}. Highest top 10 accuracy at epoch {self.highest_epoch}."
                )
                break

        self.log_print("Training completed.")
        for test, metrics in self.highest_metrics.items():
            self.log_print(
                f"{test}: Acc: {metrics['accuracy']:.4f}, Top 5: {metrics['top_5_accuracy']:.4f}, Top 10: {metrics['top_10_accuracy']:.4f}"
            )

    def run_batch(self, batch: AudioBatch, train: bool) -> tp.Dict[str, float]:
        """
        Per recording processing for training and testing. Returns average metrics
        and losses for the recording. Returns metrics on CPU.
        """

        # Some processing to ensure dims match
        brain_segments, audio_segments, recording = (
            batch.brain_segments,
            batch.audio_segments,
            batch.recording,
        )
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
        )

        # Models config decides if it is used
        conditions = {
            "study": f"{recording.study_name}",
            "subject": f"{recording.study_name}_{recording.subject_id}",
        }

        # Shuffle segments
        shuffle_indices = torch.randperm(brain_segments.shape[0])
        brain_segments, audio_segments = (
            brain_segments[shuffle_indices],
            audio_segments[shuffle_indices],
        )  # [B, C, T], [B, mel_bins, T]

        # Process by specified batch size
        batch_indices = [
            (i, min(i + self.config.batch_size, total))
            for i in range(0, total, self.config.batch_size)
        ]

        with torch.amp.autocast(dtype=self.autocast_dtype, device_type=device):

            for start, end in batch_indices:

                try:

                    if train:
                        self.optimizer.zero_grad()

                    # Slice by batch
                    brain_batch, audio_batch = (
                        brain_segments[start:end].to(self.device),
                        audio_segments[start:end].to(self.device),
                    )

                    # Forward pass
                    (output, quantizer_metrics, channel_weights, hidden_outputs) = (
                        self.model(
                            x=[brain_batch],
                            recording=[recording],
                            conditions=[conditions],
                            mel=[audio_batch],
                            train=train,
                            return_hidden_outputs=False,
                        )
                    )  # [B, C, T]
                    del channel_weights, hidden_outputs, brain_batch

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
                        self.log_print(
                            f"Loss is NaN for {recording.study_name} {recording.subject_id} {recording.session_id} {recording.task_id}."
                        )
                        missed_recordings += end - start
                        missed_batches += 1

                except Exception as e:
                    self.log_print(
                        f"Error in processing {recording.study_name} {recording.subject_id} {recording.session_id} {recording.task_id}."
                    )
                    missed_recordings += end - start
                    missed_batches += 1
                    raise e

        gc.collect()
        torch.cuda.empty_cache()

        # Correct for missed recordings and batches
        total -= missed_recordings
        batches = len(batch_indices) - missed_batches

        metrics = {
            "loss": recording_loss / batches,
            "clip_loss": recording_clip_loss / batches,
            "mse_loss": recording_mse_loss / batches,
            "commitment_loss": (recording_commitment_loss) / batches,
            "perplexity": recording_perplexity,
            "correct": recording_correct,
            "top_5_correct": recording_top_5,
            "top_10_correct": recording_top_10,
        }

        return metrics, total

    def test(self, buffer_size: int, num_workers: int, max_cache_size: int):
        """Max cache size in GB"""

        self.model.eval().to(self.device)
        self.set_seed(int(self.config.seed))
        test_start_time = time.time()

        test_datasets, test_sizes, test_dataloader = {}, {}, {}

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
                buffer_size=test_sizes[test],
                num_workers=test_sizes[test],
                max_cache_size=max_cache_size,
            )
            test_dataloader[test].start_fetching(test_datasets[test], cache=True)

        # Run tests
        for test in test_datasets.keys():
            i = 0

            all_metrics, total_batches = [], 0

            while True:

                batch = test_dataloader[test].get_recording()
                if batch is None:
                    break

                try:

                    results, num_batches = self.run_batch(batch, train=False)
                    all_metrics.append(results)
                    total_batches += num_batches

                    i += 1

                except Exception as e:
                    self.log_print(
                        f"Error in testing {test}, {batch.recording.study_name} {batch.recording.subject_id} {batch.recording.session_id} {batch.recording.task_id}. Skipping."
                    )
                    test_sizes[test] -= 1
                    continue

            # Correct -> Accuracy
            final_metrics = {
                "loss": sum([batch["loss"] for batch in all_metrics])
                / test_sizes[test],
                "clip_loss": sum([batch["clip_loss"] for batch in all_metrics])
                / test_sizes[test],
                "mse_loss": sum([batch["mse_loss"] for batch in all_metrics])
                / test_sizes[test],
                "commitment_loss": sum(
                    [batch["commitment_loss"] for batch in all_metrics]
                )
                / test_sizes[test],
                "accuracy": sum([batch["correct"] for batch in all_metrics])
                / total_batches,
                "top_5_accuracy": sum([batch["top_5_correct"] for batch in all_metrics])
                / total_batches,
                "top_10_accuracy": sum(
                    [batch["top_10_correct"] for batch in all_metrics]
                )
                / total_batches,
            }
            self.metrics["test"][test].append(final_metrics)

            self.log_print(
                f"Test {test} completed. Accuracy: {final_metrics['accuracy']:.4f}, Top 5: {final_metrics['top_5_accuracy']:.4f}, Top 10: {final_metrics['top_10_accuracy']:.4f}"
            )

            test_dataloader[test].stop()

        # Log info
        elapsed_minutes = (time.time() - test_start_time) / 60
        self.log_print(f"Testing completed in {elapsed_minutes:.2f}m.")
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

        dataloader = self.get_dataloader(buffer_size, num_workers, max_cache_size)

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
                    "highest_epoch": self.highest_epoch,
                    "highest_metrics": self.highest_metrics,
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
        config = TrainingConfigV0(
            brain_encoder_config=None, data_partition=None
        ).from_dict(config)

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
            training_session.highest_epoch = metrics.get("highest_epoch", 0)
            training_session.highest_metrics = metrics.get("highest_metrics", {})
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
