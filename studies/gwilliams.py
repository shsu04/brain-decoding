from itertools import product
import json
import os
import pandas as pd
import mne_bids
import copy
import mne
from warnings import filterwarnings

from .study import Study


class GWilliams(Study):
    def __init__(self, path: str = "data/gwilliams"):
        root_dir = os.path.join(os.getcwd(), path)
        assert os.path.exists(root_dir), f"{root_dir} does not exist"

        self.root_dir = root_dir
        self.subjects_info = pd.read_csv(
            os.path.join(self.root_dir, "participants.tsv"), sep="\t"
        )
        # Subject IDs list in string form, e.g. 00, 01, etc.
        self.subjects_list = (
            self.subjects_info["participant_id"].str.split("-").str[1].tolist()
        )

        # Task in this study is story_uid
        self.tasks = [str(i) for i in range(4)]
        self.sessions = [str(i) for i in range(2)]

        # This array will be the primary access for data
        # key is subject, values has dim (tasks, sessions)
        self.recordings = {f"{i}": [[] for _ in range(4)] for i in self.subjects_list}

        for subject, session, task in product(
            self.subjects_list, self.sessions, self.tasks
        ):
            bids_path = mne_bids.BIDSPath(
                subject=subject,
                session=session,
                task=task,
                datatype="meg",
                root=self.root_dir,
            )
            # Not all subjects did 2 sessions, but all sessions had 4 tasks
            if not bids_path.fpath.exists():
                continue

            self.recordings[subject][int(task)].append(bids_path)

        self.stimuli_type = "audio"
        self.source_link = "https://doi.org/10.1038/s41597-023-02752-5"
        super(GWilliams, self).__init__()

    def clean_recording(
        self,
        subject: str,
        task: int,
        session: int,
        n_jobs: int = None,
    ) -> tuple[mne.io.Raw, pd.DataFrame, pd.DataFrame]:
        """Returns the clean recording containing MEG channels and the relevant events.

        Arguments:
            subject -- subject ID (e.g. '01')
            task -- task number (e.g. 0)
            session -- session number (e.g. 0)

        Returns:
            tuple of:
                raw -- mne.io.Raw containing the MEG data, notch filtered at 50, 100, 150, 200, 300, 400 Hz
                word_events -- DataFrame containing the word events. Columns are 'onset', 'duration', 'word'
                sound_events -- DataFrame containing the sound events. Columns are 'onset', 'sound', 'start'
                    onset is event marker in the brain data, start is the onset in the audio file
        """

        bids_path = self.recordings[subject][task][session]
        raw = (
            mne_bids.read_raw_bids(bids_path, verbose=False)
            .pick(picks=["meg"])
            .load_data(verbose=False)
        )
        # Determined by visual inspection of the data
        raw = raw.notch_filter(
            freqs=[50, 100, 150, 200, 300, 400], verbose=False, n_jobs=n_jobs
        )

        if not hasattr(self, "old_sample_rate"):
            self.old_sample_rate = raw.info["sfreq"]
        if not hasattr(self, "info"):
            self.info = raw.info
        if not hasattr(self, "channel_names"):
            self.channel_names = [
                channel for channel in raw.ch_names if any(["MEG" in channel])
            ]

        annotations_df = pd.DataFrame(raw.annotations)
        word_events, sound_events = copy.deepcopy(annotations_df), copy.deepcopy(
            annotations_df
        )

        # Filter out events description to only contain pronounced word
        word_events = word_events[
            word_events["description"].str.contains("'kind': 'word'")
            & word_events["description"].str.contains("'pronounced': 1.0")
        ].reset_index(drop=True)

        word_events["word"] = word_events["description"].apply(
            lambda x: json.loads(x.replace("'", '"'))["word"]
        )
        word_events.drop(["description", "orig_time"], axis=1, inplace=True)

        # Filter out events description to only contain sound and its onset in audio file
        sound_events = sound_events[
            sound_events["description"].str.contains("'kind': 'sound'")
        ].reset_index(drop=True)
        sound_events["sound"] = sound_events["description"].apply(
            lambda x: json.loads(x.replace("'", '"'))["sound"]
        )
        # sound_events = sound_events.drop_duplicates(subset="onset", keep="last")
        sound_events["start"] = sound_events["description"].apply(
            lambda x: json.loads(x.replace("'", '"'))["start"]
        )
        sound_events.drop(
            ["description", "orig_time", "duration"], axis=1, inplace=True
        )

        return raw, word_events, sound_events
