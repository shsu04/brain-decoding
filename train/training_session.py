from abc import ABC, abstractmethod
import gc
import random
import time
from numpy import record
from tqdm import tqdm
import numpy as np
from config.simpleconv_config import SimpleConvConfig
from models.simpleconv import SimpleConv
from studies.study_factory import StudyFactory
import typing as tp
import json
from itertools import product
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import os
import logging
import shutil
import torch
from config import SimpleConvConfig, Config, TrainingConfig


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

        assert len(studies) > 0, "At least one study root path must be provided"
        assert all(
            os.path.exists(data_path + "/" + study) for study in studies
        ), "All study root paths must exist"        
        os.makedirs(save_path, exist_ok=True)

        logging.basicConfig(
            filename=os.path.join(save_path, "training_log.log"),
            level=logging.INFO,
            format="%(asctime)s %(message)s",
            filemode="w",
        )
        self.logger = logging.getLogger()

        self.config = config
        self.data_path = data_path
        self.save_path = save_path

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
        
        self.dataloader, self.test_dataoader = None, {}

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
        np.random.seed(seed)
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
                        self.dataset["test"]["unseen_task"].append(recording)
                # Unseen subject and train
                else:
                    if task in data_partition["testing_tasks"]:
                        self.dataset["test"]["unseen_subject"].append(recording)
                    else:
                        self.dataset["train"].append(recording)

        self.log_print(f"Data partitioned on studies {list(self.studies.keys())}.")
        self.log_print(
            f"Train: {len(self.dataset['train'])}, Unseen Task: {len(self.dataset['test']['unseen_task'])}, Unseen Subject: {len(self.dataset['test']['unseen_subject'])}, Unseen Both: {len(self.dataset['test']['unseen_both'])}.\n"
        )

    def discard_nan(self):
        # Consider putting this, generating conditional layer (subject and dataset) in dataloader
        pass

    def log_print(self, message):
        print(message)
        self.logger.info(message)
