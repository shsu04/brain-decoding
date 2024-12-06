from email.mime import audio
import ray
from typing import Dict, List, Tuple, Optional
import torch
from queue import Queue
import threading
from dataclasses import dataclass
import numpy as np
from studies import Study
from utils.fetch import fetch_audio_and_brain_pairs


@dataclass
class DataBatch:
    brain_segments: Dict[str, torch.Tensor]
    audio_segments: torch.Tensor
    layout: torch.Tensor
    metadata: Dict[str, int]  # Contains study, subject, task, session info


@ray.remote
class DataFetcher:
    """Fetches a single recording from an audio and brain pair dataset."""

    def __init__(
        self,
        studies: dict[str, Study],
        pre_processor,
        max_random_shift: float,
        window_size: int,
        window_stride: int,
        baseline_window: float,
        frequency_bands: dict,
        brain_clipping: int,
        new_freq: int,
        notch_filter: bool,
        audio_sample_rate: int,
        hop_length: int,
        audio_processor: str,
        n_jobs: int = 1,
    ):
        self.studies = studies
        self.pre_processor = pre_processor
        self.max_random_shift = max_random_shift
        self.window_size = window_size
        self.window_stride = window_stride
        self.baseline_window = baseline_window
        self.frequency_bands = frequency_bands
        self.brain_clipping = brain_clipping
        self.notch_filter = notch_filter
        self.new_freq = new_freq
        self.audio_sample_rate = audio_sample_rate
        self.audio_processor = audio_processor
        self.hop_length = hop_length
        self.n_jobs = n_jobs

    def fetch_recording(
        self, study_name: str, subject: int, task: int, session: int
    ) -> DataBatch:

        brain_segments, audio_segments, layout = fetch_audio_and_brain_pairs(
            pre_processor=self.pre_processor,
            # ID
            study=self.studies[study_name],
            subject=subject,
            task=task,
            session=session,
            # Brain Param
            max_random_shift=self.max_random_shift,
            window_size=self.window_size,
            window_stride=self.window_stride,
            baseline_window=self.baseline_window,
            frequency_bands=self.frequency_bands,
            new_freq=self.new_freq,
            brain_clipping=self.brain_clipping,
            notch_filter=self.notch_filter,
            # Audio param
            audio_sample_rate=self.audio_sample_rate,
            hop_length=self.hop_length,
            audio_processor=self.audio_processor,
            n_jobs=self.n_jobs,
        )

        return DataBatch(
            brain_segments=brain_segments,
            audio_segments=audio_segments,
            layout=layout,
            metadata={
                "study": study_name,
                "subject": subject,
                "task": task,
                "session": session,
            },
        )


class ParallelDataLoader:
    def __init__(
        self,
        # Param
        buffer_size: int,
        num_workers: int,
        studies: dict[str, Study],
        pre_processor,
        # Brain Param
        max_random_shift: float,
        window_size: int,
        window_stride: int,
        baseline_window: float,
        frequency_bands: dict,
        brain_clipping: int,
        new_freq: int,
        notch_filter: bool,
        # Audio param
        audio_sample_rate: int,
        hop_length: int,
        audio_processor: str,
        n_jobs: int = 1,
    ):
        """Initialize the parallel data loader with Ray workers.

        Args:
            buffer_size: Number of batches to pre-fetch
            num_workers: Number of Ray workers to use for parallel fetching
        """
        if not ray.is_initialized():
            ray.init()

        self.buffer_size = buffer_size
        self.queue = Queue(maxsize=buffer_size)
        self.stop_event = threading.Event()

        # Create pool of workers
        self.fetchers = [
            DataFetcher.remote(
                studies=studies,
                pre_processor=pre_processor,
                max_random_shift=max_random_shift,
                window_size=window_size,
                window_stride=window_stride,
                baseline_window=baseline_window,
                frequency_bands=frequency_bands,
                new_freq=new_freq,
                brain_clipping=brain_clipping,
                notch_filter=notch_filter,
                audio_sample_rate=audio_sample_rate,
                hop_length=hop_length,
                audio_processor=audio_processor,
                n_jobs=n_jobs,
            )
            for _ in range(num_workers)
        ]

        self.current_worker = 0
        self.fetch_thread = None
        self.pending_futures = []

    def start_fetching(self, batch_indices: List[Tuple[int, int, int, int]]):
        """Start background fetching of data batches.

        Args:
            batch_indices: List of (study, subject, task, session) tuples to fetch
        """

        def _fetch_worker():
            for study_name, subject, task, session in batch_indices:

                if self.stop_event.is_set():
                    break

                # Circular queue
                worker = self.fetchers[self.current_worker]
                self.current_worker = (self.current_worker + 1) % len(self.fetchers)

                # launch fetch and store future
                future = worker.fetch_recording.remote(
                    study_name, subject, task, session
                )
                self.pending_futures.append(future)

                # Wait if buffer full
                while len(self.pending_futures) > self.buffer_size:
                    done_futures, self.pending_futures = ray.wait(
                        self.pending_futures, num_returns=1, timeout=None
                    )
                    batch = ray.get(done_futures[0])
                    self.queue.put(batch)

            # Wait for remaining futures
            for future in self.pending_futures:
                batch = ray.get(future)
                self.queue.put(batch)

        self.fetch_thread = threading.Thread(target=_fetch_worker)
        self.fetch_thread.start()

    def get_recording(self) -> Optional[DataBatch]:
        """Get the next recording from the queue.

        Returns:
            DataBatch or None if queue is empty and fetching is complete
        """
        if self.queue.empty() and not self.fetch_thread.is_alive():
            return None
        return self.queue.get()

    def stop(self):
        """Stop the background fetching thread."""
        self.stop_event.set()
        if self.fetch_thread and self.fetch_thread.is_alive():
            self.fetch_thread.join()
