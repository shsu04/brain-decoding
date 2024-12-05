from itertools import product
import json
import os
import pandas as pd
import mne_bids
import copy
import mne
from warnings import filterwarnings
import shutil

from studies.study import Study

filterwarnings("ignore")


class Schoffelen(Study):

    def __init__(self, path: str = "data/schoffelen"):
        root_dir = os.path.join(os.getcwd(), path)
        assert os.path.exists(root_dir), f"{root_dir} does not exist"

        self.root_dir = root_dir
        self.cache_dir = os.path.join(os.getcwd(), "cache", "schoffelen")

        # Clear cache
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)

        # Create cache directory
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        self.subjects_info = pd.read_csv(
            os.path.join(self.root_dir, "participants.tsv"), sep="\t"
        )
        # Subject IDs list in string form, e.g. 000, 001, etc.
        self.subjects_list = (
            self.subjects_info["participant_id"].str.split("-").str[1].tolist()
        )

        self.sessions = ["compr"]  # Not taking empty room
        self.tasks = [
            "001",
            "002",
            "003",
            "004",
            "005",
            "006",
            "007",
            "008",
            "009",
            "010",
        ]

        self.recordings = [
            [[] for _ in range(len(self.tasks))] for i in range(len(self.subjects_list))
        ]

        for subject, task, session in product(
            [i for i in range(len(self.subjects_list))],
            [i for i in range(len(self.tasks))],
            [i for i in range(len(self.sessions))],
        ):
            # Task and session are swapped in the BIDSPath, since in
            # this study the naming convention is swapped

            bids_path = mne_bids.BIDSPath(
                subject=self.subjects_list[subject],
                session=self.tasks[task],
                task=self.sessions[session],
                root=self.root_dir,
            )

            if not bids_path.fpath.exists():
                raise FileNotFoundError(f"{bids_path.fpath} does not exist")

            self.recordings[subject][task].append(bids_path)

        # The only valid channel types in this study
        self.types = [
            "MLC",
            "MLF",
            "MLO",
            "MLP",
            "MLT",
            "MRC",
            "MRF",
            "MRO",
            "MRP",
            "MRT",
            "MZC",
            "MZF",
            "MZO",
            "MZP",
        ]

        self.stimuli_type = "audio"
        self.source_link = "https://www.nature.com/articles/s41597-022-01382-7"
        super(Schoffelen, self).__init__()

    def clean_recording(
        self,
        subject: int,
        task: int,
        session,
        notch_filter: bool = False,
        n_jobs: int = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Returns the clean recording containing MEG channels and the relevant events.

        Arguments:
            subject -- subject number (e.g. 0)
            task -- task number (e.g. 0)
            session -- session number (e.g. 0)
            notch_filter -- whether to apply notch filter

        Returns:
            tuple of:
                raw -- mne.io.Raw containing the MEG data, notch filtered at 50, 100, 150, 200, 300, 400 Hz
                word_events -- DataFrame containing the word events. Columns are 'onset', 'duration', 'word'
                sound_events -- DataFrame containing the sound events. Columns are 'onset', 'sound', 'end'
                    onset is event marker in the brain data, start is the onset in the audio file
        """
        bids_path = self.recordings[subject][task][session]
        raw = mne.io.read_raw(bids_path, verbose=False)

        if not hasattr(self, "old_sample_rate"):
            self.old_sample_rate = raw.info["sfreq"]
        if not hasattr(self, "info"):
            self.info = raw.info
        # Not all channels are relevant in this study
        if not hasattr(self, "channel_names"):
            self.channel_names = [
                channel
                for channel in raw.ch_names
                if any([type in channel for type in self.types])
            ]

        # Filter to only contain relevant channels
        raw = raw.pick(picks=self.channel_names, verbose=False)
        raw = raw.load_data(verbose=False)

        if notch_filter:
            # Determined by visual inspection of the data, exclude powerline noise
            raw = raw.notch_filter(
                freqs=[50, 100, 150, 200, 300, 400], verbose=False, n_jobs=n_jobs
            )

        # We do this since this study does not have complete events from the mne.Raw object
        events_path = bids_path.copy().update(suffix="events", extension=".tsv")
        annotations_df = pd.read_csv(
            str(events_path.directory) + "/meg/" + events_path.basename, sep="\t"
        )
        word_events, sound_events = copy.deepcopy(annotations_df), copy.deepcopy(
            annotations_df
        )

        # Filter out phonemes and other irrelevant annotations
        word_events = word_events[word_events["type"].str.contains("word_onset")].drop(
            columns=["sample", "type"]
        )

        # Rename to match GWilliams
        word_events.columns = ["onset", "duration", "word"]

        word_events["word"] = word_events["word"].apply(lambda x: x.lower())
        # 'sp' is recurrent, but author did not provide explaination
        word_events = word_events[word_events["word"] != "sp"].reset_index(drop=True)

        # Start of audio and keep only valid audio files
        sound_events = sound_events[
            (sound_events["type"].str.contains("wav_onset"))
            & (sound_events["value"] != str(100))
        ].drop(columns=["duration", "sample", "type"])
        # Phoneme onset where the next row is wav onset (end time of audio file)
        end_times = annotations_df[
            annotations_df["type"].str.contains("phoneme_onset")
            & annotations_df["type"].shift(-1).str.contains("wav_onset")
        ]
        # So that columns are onset, sound, end
        sound_events["end"] = end_times["onset"].values
        # stimuli/{tasknum}_{num}.wav is the path to the audio file
        sound_events["sound"] = sound_events["value"].apply(
            lambda x: f"stimuli/{self.tasks[task][1:]}_{x[0]}.wav"
        )
        sound_events = sound_events.drop(columns=["value"])

        return raw, word_events, sound_events
