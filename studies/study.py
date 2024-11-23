from abc import ABC, abstractmethod
import pandas as pd
import mne


class Study(ABC):
    """Abstract class for a study. Accessed by the pre-processor. All
    implementations of a study must provide a list of subjects and recordings.
    Recordings must be a list of dictionaries, where each dictionary has a hierarchy
    as subject -> task -> session -> BIDSPath, stored as a dictionary where key is
    subject in string form (eg 01) and value is a list (task) of lists (session) of BIDSPath.
    """

    def __init__(self):
        super(Study, self).__init__()

        if not hasattr(self, "root_dir"):
            raise AttributeError("Study must have a root_dir attribute.")
        if not hasattr(self, "subjects_info"):
            raise AttributeError("Study must have a subjects_info attribute.")
        if not hasattr(self, "subjects_list"):
            raise AttributeError("Study must have a subjects_list attribute.")
        if not hasattr(self, "tasks"):
            raise AttributeError("Study must have a tasks attribute.")
        if not hasattr(self, "sessions"):
            raise AttributeError("Study must have a sessions attribute.")
        if not hasattr(self, "recordings"):
            raise AttributeError("Study must have a recordings attribute.")
        if not hasattr(self, "stimuli_type"):
            raise AttributeError("Study must have a stimuli type attribute.")
        if not hasattr(self, "source_link"):
            raise AttributeError("Study must have a source_link attribute.")

    @abstractmethod
    def clean_recording(
        self,
        subject: str,
        task: int,
        session: int,
    ) -> tuple[mne.io.Raw, dict[str, pd.DataFrame]]:
        """Returns the recording for a given subject, task, and session as a
        mne.io.Raw object and a dictionary of relevant events for each channel.

        1. mne.io.Raw object, notch filtered for the study's power line frequency
        2. df containing at least the onset column
        """
        pass
