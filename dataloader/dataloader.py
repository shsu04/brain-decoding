"""
General parallel data loader for any DataBatch type.
"""

from calendar import c
import os
import threading
from queue import Queue
from typing import Optional
from librosa import ex
import ray
import multiprocessing


from .batch import Batch
from .audio_batch import AudioBatchFetcher
from studies import Recording

# String names have to match recording types
BATCHTYPES = {
    "audio": AudioBatchFetcher,
}


class BatchFetcherFactory:
    @staticmethod
    def create_batch_fetcher(batch_type: str, **kwargs):
        if batch_type not in BATCHTYPES:
            raise ValueError(f"Batch type {batch_type} not found")
        return BATCHTYPES[batch_type].remote(**kwargs)


class DataLoader:
    def __init__(
        self,
        buffer_size: int,
        max_cache_size_gb: float,
        cache_dir: str,
        # Common brain params shared by all batch types
        notch_filter: bool,
        frequency_bands: dict[str, tuple[float, float]],
        scaling: str,
        brain_clipping: int,
        baseline_window: float,
        new_freq: int,
        # Specifies batch type creation
        batch_types: dict[str, int],
        batch_kwargs: dict[str, dict],
    ):
        """Initialize the parallel data loader with Ray workers.

        Args:
            buffer_size -- size of the queue to store fetched data
            max_cache_size_gb -- maximum size of the cache in bytes
            cache_dir -- directory to store the cache
            notch_filter -- whether to apply notch filter to the raw data to remove powerline
            frequency_bands -- dictionary of frequency bands tuple,
                brain segements will be returned for each band in the dictionary
            scaling -- scaling method to apply to the brain data
            brain_clipping -- standard deviation to clip the brain data to
            baseline_window -- window size to use for baseline normalization
            new_freq -- new frequency to resample the brain data to

            batch_types -- dictionary of batch types to create and their counts
            batch_kwargs -- dictionary of batch type kwargs
        """
        # Check if valid batch types and kwargs
        if batch_types == {} or batch_kwargs == {}:
            raise ValueError(
                "batch_types and batch_kwargs must have at least one element"
            )
        if not all([k in BATCHTYPES for k in batch_types.keys()]):
            raise ValueError("batch_types must be a subset of BATCHTYPES")
        if not all([k in BATCHTYPES for k in batch_kwargs.keys()]):
            raise ValueError("batch_kwargs must be a subset of BATCHTYPES")
        if batch_types.keys() != batch_kwargs.keys():
            raise ValueError("batch_types and batch_kwargs keys must match")
        assert buffer_size > 0, "Buffer size must be greater than 0"
        assert max_cache_size_gb > 30, "Max cache size must be greater than 30"
        # Must've been created by the study
        assert os.path.exists(cache_dir), "Cache directory does not exist"

        # If batch sizes total workers exceed cpu, reduce proportionally
        actual_workers, requested_workers = multiprocessing.cpu_count(), sum(
            batch_types.values()
        )
        if requested_workers > actual_workers:
            ratio = actual_workers / requested_workers
            print(
                f"Total batch workers exceed resources. Reducing by ratio {ratio:.2f}"
            )
            for k in batch_types.keys():
                batch_types[k] = int(batch_types[k] * ratio)

        self.buffer_size = buffer_size
        self.queue = Queue(maxsize=buffer_size)
        self.stop_event = threading.Event()
        self.max_cache_size_gb = max_cache_size_gb
        self.cache_dir = cache_dir

        # Dictionary of batch type to list of fetchers
        self.fetchers = {}

        # Create pool of workers
        for batch_type, count in batch_types.items():

            batch_fetchers = []

            for _ in range(count):
                batch_fetchers.append(
                    BatchFetcherFactory.create_batch_fetcher(
                        batch_type,
                        notch_filter=notch_filter,
                        frequency_bands=frequency_bands,
                        scaling=scaling,
                        brain_clipping=brain_clipping,
                        baseline_window=baseline_window,
                        new_freq=new_freq,
                        **batch_kwargs[batch_type],
                    )
                )

            self.fetchers[batch_type] = batch_fetchers

        self.current_worker = 0
        self.fetch_thread = None
        self.pending_futures = []

    def start_fetching(self, recordings: list[Recording], cache: bool):
        """Start the background fetching thread.

        Args:
            recordings: List of recordings to fetch. The batch type is determiend
                by recording.type parameter.
        """

        def _fetch_worker():

            for recording in recordings:

                if self.stop_event.is_set():
                    break

                if recording.type not in self.fetchers.keys():
                    print(
                        f"Recording type {recording.type} not found. Skipping {recording.cache_path}"
                    )
                    continue

                # Check cache size and adjust the cache flag
                current_cache_size = self.get_approximate_cache_size(self.cache_dir)
                cache_flag = cache

                if current_cache_size >= self.max_cache_size_gb:
                    cache_flag = False

                try:
                    # Circular queue
                    worker = self.fetchers[recording.type][self.current_worker]
                    self.current_worker = (self.current_worker + 1) % len(
                        self.fetchers[recording.type]
                    )
                    # launch fetch and store future
                    future = worker.fetch.remote(recording, cache_flag)
                    self.pending_futures.append(future)
                except ValueError as e:
                    # If message same as "...", print and skip
                    if "Number of brain and audio windows do not match" in str(e):
                        print(f"Recording not found. Skipping {recording.cache_path}.")
                    else:
                        print(f"Error fetching {recording.cache_path}: {e}")
                except Exception as e:
                    print(f"Error fetching {recording.cache_path}: {e}")
                    continue

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

    def get_recording(self) -> Optional[Batch]:
        """Get the next recording from the queue, regardless of type.

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

    def get_approximate_cache_size(self, cache_dir: str) -> float:
        """Quick estimate of cache size in GB. Not thread safe, but fast."""
        try:
            total_size = sum(
                os.path.getsize(os.path.join(dirpath, f))
                for dirpath, _, filenames in os.walk(cache_dir)
                for f in filenames
            )
            return total_size / (1024**3)  # Convert to GB
        except:
            return 0  # If error reading size, assume safe to continue
