from abc import ABC, abstractmethod
import random
import numpy as np
from studies.study_factory import StudyFactory
import typing as tp
from itertools import product
import os
import logging
import shutil
import torch
from config import TrainingConfig


class TrainingSession(ABC):
    def __init__(
        self,
        config: TrainingConfig,
        studies: tp.Dict[str, str],
        data_path: str = "/home/ubuntu/brain-decoding/data",
        save_path: str = "/home/ubuntu/brain-decoding/saves",
        clear_cache: bool = False,
        cache_enabled: bool = True,
        max_cache_size: int = 100,
        cache_name: str = "cache",
        download_studies: bool = False,
    ):
        """Initializes a training session with the provided configuration and data.

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

        base_name = os.path.basename(save_path)
        if base_name.startswith("epoch_"):
            # Then we just take the parent directory
            save_path = os.path.dirname(save_path)
            
        assert len(studies) > 0, "At least one study root path must be provided"
        os.makedirs(save_path, exist_ok=True)

        # Batch logger
        general_logger = logging.getLogger(f"general_logger_{save_path}")
        general_logger.setLevel(logging.INFO)

        # Remove any existing handlers from the logger so we can re-add in append mode
        for h in list(general_logger.handlers):
            general_logger.removeHandler(h)
            h.close()

        # Now add a fresh handler in append mode
        fh = logging.FileHandler(
            os.path.join(save_path, "training_log.log"), mode="a"  # <-- 'a' for append
        )
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        general_logger.addHandler(fh)
        general_logger.propagate = False
        
        # Epoch Results Logger
        epoch_logger = logging.getLogger(f"epoch_logger_{save_path}")
        epoch_logger.setLevel(logging.INFO)

        for h in list(epoch_logger.handlers):
            epoch_logger.removeHandler(h)
            h.close()

        fh_epoch = logging.FileHandler(os.path.join(save_path, "epoch_log.log"), mode="a")
        fh_epoch.setFormatter(logging.Formatter("%(message)s"))
        epoch_logger.addHandler(fh_epoch)
        epoch_logger.propagate = False
        
        self.logger = general_logger
        self.epoch_logger = epoch_logger

        self.config = config
        self.data_path = data_path
        self.save_path = save_path
        self.cache_name = cache_name

        # Create studies accessor
        self.studies = {}
        for study, batch_type in studies.items():
            path = os.path.join(data_path, study)
            try:
                self.studies[study] = StudyFactory.create_study(
                    study_name=study,
                    batch_type=batch_type,
                    path=path,
                    cache_enabled=cache_enabled,
                    max_cache_size=max_cache_size,
                    cache_name=cache_name,
                    download=download_studies,
                )
                if clear_cache:
                    shutil.rmtree(self.studies[study].cache_dir)
                    os.makedirs(self.studies[study].cache_dir)
                    self.log_print(f"Cleared cache for study {study}")
            except ValueError as e:
                self.log_print(f"Error loading study {study}: {e}")

        self.dataset = {
            "train": [],
            "test": {
                "unseen_subject": [],
                "unseen_task": [],
                "unseen_both": [],
            },
        }

        self.partition_datasets()

        self.metrics = {
            "train": [],
            "test": {
                "unseen_subject": [],
                "unseen_task": [],
                "unseen_both": [],
            },
        }

        self.error = None
        self.set_seed(int(self.config.seed))

        # Set conditions
        if self.config.brain_encoder_config.conditions is not None:
            if "study" in self.config.brain_encoder_config.conditions:
                self.config.brain_encoder_config.conditions["study"] = list(
                    studies.keys()
                )

            if "subject" in self.config.brain_encoder_config.conditions:
                subjects = set()
                for recording in self.recordings:
                    subjects.add(f"{recording.study_name}_{recording.subject_id}")
                self.config.brain_encoder_config.conditions["subject"] = list(subjects)

        # Check if GPU is NVIDIA V100, A100, or H100
        torch.set_float32_matmul_precision("high")
        gpu_ok = False
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            if major >= 8:
                gpu_ok = True

        # Compile if on NVIDIA V100, A100, or H100 for faster training
        if not gpu_ok:
            self.log_print(
                "GPU is not Volta, Ampere, or Hopper architecture. Speedup numbers may be lower "
                "than expected without bf16. Training at fp32."
            )
            self.autocast_dtype = torch.float32
        else:
            self.autocast_dtype = torch.bfloat16

        # Metric for early stopping
        self.highest_epoch, self.highest_metrics, self.highest_average_test_accuracy = (
            0,
            None,
            0,
        )

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def run_batch(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def save(self):
        pass

    def pre_process_all_recordings(self):
        pass

    def set_seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def partition_datasets(self):
        """
        Partitions the data into training and various testing sets, based on
        the named holdout sessions and tasks specified in TrainingConfig
        """
        self.recordings = []

        for study_name, study in self.studies.items():
            if study_name not in self.config.data_partition.keys():
                raise ValueError(f"Study {study_name} not found in data partition")

            data_partition = self.config.data_partition[study_name]

            for subject, session, task in product(
                [i for i in range(len(study.subjects))],
                [i for i in range(len(study.sessions))],
                [i for i in range(len(study.tasks))],
            ):
                # If recording exists
                if study.recording_exists(
                    subject=study.subjects[subject],
                    session=study.sessions[session],
                    task=study.tasks[task],
                ):
                    recording = study.recordings[subject][session][task]
                    self.recordings.append(recording)
                else:
                    continue

                # Unseen both and task
                if subject in data_partition["testing_subjects"]:
                    if task in data_partition["testing_tasks"]:
                        self.dataset["test"]["unseen_both"].append(recording)
                    else:
                        self.dataset["test"]["unseen_subject"].append(recording)
                # Unseen subject and train
                else:
                    if task in data_partition["testing_tasks"]:
                        self.dataset["test"]["unseen_task"].append(recording)
                    else:
                        self.dataset["train"].append(recording)

        self.log_print(f"Data partitioned on studies {list(self.studies.keys())}.")
        self.log_print(
            f"Train: {len(self.dataset['train'])}, Unseen Task: {len(self.dataset['test']['unseen_task'])}, Unseen Subject: {len(self.dataset['test']['unseen_subject'])}, Unseen Both: {len(self.dataset['test']['unseen_both'])}.\n"
        )

    def discard_nan(self):
        # Consider putting this, generating conditional layer (subject and dataset) in dataloader
        pass

    def log_print(self, message: str):
        print(message)
        self.logger.info(message)
        self.epoch_logger.info(message)

    def log_no_print(self, message: str):
        self.logger.info(message)
        self.epoch_logger.info(message)

    def delete_subdirectories(self, save_path):
        """Recursively deletes all subdirectories in the given save_path."""
        if not os.path.exists(save_path):
            return  # Nothing to delete if the path doesn't exist

        for entry in os.listdir(save_path):
            entry_path = os.path.join(save_path, entry)
            if os.path.isdir(entry_path):
                try:
                    shutil.rmtree(entry_path)
                    print(f"Deleted directory: {entry_path}")
                except Exception as e:
                    print(f"Error deleting {entry_path}: {e}")
