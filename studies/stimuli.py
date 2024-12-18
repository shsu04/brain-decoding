import threading
import os

import librosa
import numpy as np


class Stimuli:
    """
    Thread-safe central manager, each study holds an instance of this class to avoid
    deadlock when each recording fetches stimuli, and to avoid duplicate caching.
    """

    def __init__(
        self,
        root_path: str,
        names: list[str],
        cache_enabled: bool = True,
        max_cache_size: int = 100,
    ):
        """
        Arguments:
            root_path -- root path shared by all stimuli in the study
            names -- list of names of the stimuli
        """
        assert os.path.exists(root_path), f"Path {root_path} does not exist."

        self.root_path = root_path
        self.names = names
        self.lock = threading.Lock()

        if cache_enabled:
            self.cache_enabled = True
            self.cache = {}
            self.max_cache_size = max_cache_size
        else:
            self.cache_enabled = False

    def enable_cache(self, max_cache_size: int = 100):
        """
        Enable caching of stimuli audio.

        Arguments:
            max_cache_size -- maximum number of items to cache
        """
        self.cache_enabled = True
        self.cache = {} if not self.cache else self.cache
        self.max_cache_size = max_cache_size

    def empty_cache(self):
        """
        Empty the cache.
        """
        self.cache = {}

    def __getstate__(self):
        """Return the state for pickling, excluding the lock"""
        state = self.__dict__.copy()
        # Remove the unpickleable lock
        state.pop("lock", None)
        return state

    def __setstate__(self, state):
        """Restore the state and recreate the lock"""
        self.__dict__.update(state)
        # Recreate the lock
        self.lock = threading.Lock()

    def load_audio(self, names: list[str]) -> dict[str, np.ndarray]:
        """
        Fetch the audio from the cache if it exists, otherwise load it from the disk.

        Arguments:
            names -- names of the audio file to load. Each must be in the list of stimuli names.
                    In addition, root_path/name must be the target file.

        Returns:
            audios -- audio data as a dict of numpy arrays, dim = (seconds * sample_rate,)
        """
        with self.lock:

            audios = {}

            for name in names:
                if name not in self.names:
                    raise ValueError(
                        f"Stimuli name {name} not found in the list of stimuli."
                    )

                # Check cache
                if self.cache_enabled and name in self.cache:
                    return self.cache[name]

                audio_path = f"{self.root_path}/{name}"

                # Stimuli event names sometimes need to be adjusted in the study
                # load events function for this to work
                if not os.path.exists(audio_path):
                    raise FileNotFoundError(
                        f"Stimuli {name} not found in {self.root_path}"
                    )

                audio, _ = librosa.load(audio_path, sr=16000)
                audio = audio.astype(np.float32)

                # If caching is enabled, add to cache.
                if self.cache_enabled:
                    if len(self.cache) >= self.max_cache_size:
                        self.cache.popitem()
                    self.cache[name] = audio

                audios[name] = audio

            return audios
