from abc import ABC, abstractmethod


class TrainingSession(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def run_recording(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass

    def pre_process_all_recordings(self):
        pass

    def check_studies_batch_fetcher_valid(self):
        pass

    def partition_datasets(self):
        pass

    def discard_nan(self):
        # Consider putting this, generating conditional layer (subject and dataset) in dataloader
        pass

    def log_print(self, message):
        pass
