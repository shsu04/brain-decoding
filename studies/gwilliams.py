from itertools import product
import json
import os
import pandas as pd
import mne_bids
import copy
import mne
from warnings import filterwarnings
import shutil
import pathlib
import librosa
import numpy as np

from .study import Study

filterwarnings("ignore")


class GWilliams(Study):
    def __init__(self, path: str = "data/gwilliams"):
        root_dir = os.path.join(os.getcwd(), path)
        assert os.path.exists(root_dir), f"{root_dir} does not exist"

        self.root_dir = root_dir
        self.cache_dir = os.path.join(os.getcwd(), "cache", "gwilliams")

        # Create cache directory
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

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

        # Recordings is a 3D array, where the first dimension is the subject,
        # the second dimension is the task, and the third dimension is the session.
        self.recordings = [
            [[] for _ in range(len(self.tasks))] for i in range(len(self.subjects_list))
        ]

        for subject, task, session in product(
            [i for i in range(len(self.subjects_list))],
            [i for i in range(len(self.tasks))],
            [i for i in range(len(self.sessions))],
        ):
            bids_path = mne_bids.BIDSPath(
                subject=self.subjects_list[subject],
                session=self.sessions[session],
                task=self.tasks[task],
                datatype="meg",
                root=self.root_dir,
            )
            # Not all subjects did 2 sessions, but all sessions had 4 tasks
            if not bids_path.fpath.exists():
                continue

            self.recordings[subject][task].append(bids_path)

        self.channel_names = [
            "MEG 001",
            "MEG 002",
            "MEG 003",
            "MEG 004",
            "MEG 005",
            "MEG 006",
            "MEG 007",
            "MEG 008",
            "MEG 009",
            "MEG 010",
            "MEG 011",
            "MEG 012",
            "MEG 013",
            "MEG 014",
            "MEG 015",
            "MEG 016",
            "MEG 017",
            "MEG 018",
            "MEG 019",
            "MEG 020",
            "MEG 021",
            "MEG 022",
            "MEG 023",
            "MEG 024",
            "MEG 025",
            "MEG 026",
            "MEG 027",
            "MEG 028",
            "MEG 029",
            "MEG 030",
            "MEG 031",
            "MEG 032",
            "MEG 033",
            "MEG 034",
            "MEG 035",
            "MEG 036",
            "MEG 037",
            "MEG 038",
            "MEG 039",
            "MEG 040",
            "MEG 041",
            "MEG 042",
            "MEG 043",
            "MEG 044",
            "MEG 045",
            "MEG 046",
            "MEG 047",
            "MEG 048",
            "MEG 049",
            "MEG 050",
            "MEG 051",
            "MEG 052",
            "MEG 053",
            "MEG 054",
            "MEG 055",
            "MEG 056",
            "MEG 057",
            "MEG 058",
            "MEG 059",
            "MEG 060",
            "MEG 061",
            "MEG 062",
            "MEG 063",
            "MEG 064",
            "MEG 065",
            "MEG 066",
            "MEG 067",
            "MEG 068",
            "MEG 069",
            "MEG 070",
            "MEG 071",
            "MEG 072",
            "MEG 073",
            "MEG 074",
            "MEG 075",
            "MEG 076",
            "MEG 077",
            "MEG 078",
            "MEG 079",
            "MEG 080",
            "MEG 081",
            "MEG 082",
            "MEG 083",
            "MEG 084",
            "MEG 085",
            "MEG 086",
            "MEG 087",
            "MEG 088",
            "MEG 089",
            "MEG 090",
            "MEG 091",
            "MEG 092",
            "MEG 093",
            "MEG 094",
            "MEG 095",
            "MEG 096",
            "MEG 097",
            "MEG 098",
            "MEG 099",
            "MEG 100",
            "MEG 101",
            "MEG 102",
            "MEG 103",
            "MEG 104",
            "MEG 105",
            "MEG 106",
            "MEG 107",
            "MEG 108",
            "MEG 109",
            "MEG 110",
            "MEG 111",
            "MEG 112",
            "MEG 113",
            "MEG 114",
            "MEG 115",
            "MEG 116",
            "MEG 117",
            "MEG 118",
            "MEG 119",
            "MEG 120",
            "MEG 121",
            "MEG 122",
            "MEG 123",
            "MEG 124",
            "MEG 125",
            "MEG 126",
            "MEG 127",
            "MEG 128",
            "MEG 129",
            "MEG 130",
            "MEG 131",
            "MEG 132",
            "MEG 133",
            "MEG 134",
            "MEG 135",
            "MEG 136",
            "MEG 137",
            "MEG 138",
            "MEG 139",
            "MEG 140",
            "MEG 141",
            "MEG 142",
            "MEG 143",
            "MEG 144",
            "MEG 145",
            "MEG 146",
            "MEG 147",
            "MEG 148",
            "MEG 149",
            "MEG 150",
            "MEG 151",
            "MEG 152",
            "MEG 153",
            "MEG 154",
            "MEG 155",
            "MEG 156",
            "MEG 157",
            "MEG 158",
            "MEG 159",
            "MEG 160",
            "MEG 161",
            "MEG 162",
            "MEG 163",
            "MEG 164",
            "MEG 165",
            "MEG 166",
            "MEG 167",
            "MEG 168",
            "MEG 169",
            "MEG 170",
            "MEG 171",
            "MEG 172",
            "MEG 173",
            "MEG 174",
            "MEG 175",
            "MEG 176",
            "MEG 177",
            "MEG 178",
            "MEG 179",
            "MEG 180",
            "MEG 181",
            "MEG 182",
            "MEG 183",
            "MEG 184",
            "MEG 185",
            "MEG 186",
            "MEG 187",
            "MEG 188",
            "MEG 189",
            "MEG 190",
            "MEG 191",
            "MEG 192",
            "MEG 193",
            "MEG 194",
            "MEG 195",
            "MEG 196",
            "MEG 197",
            "MEG 198",
            "MEG 199",
            "MEG 200",
            "MEG 201",
            "MEG 202",
            "MEG 203",
            "MEG 204",
            "MEG 205",
            "MEG 206",
            "MEG 207",
            "MEG 208",
        ]

        self.audio_cache = {}
        # Search for all audio files in the root directory and its subdirectories
        wav_files = list(pathlib.Path(self.root_dir).rglob("*.wav"))
        for wav_file in wav_files:
            audio, _ = librosa.load(wav_file, sr=16000)
            audio = audio.astype(np.float32)
            self.audio_cache[f"stimuli/audio/{wav_file.name}"] = audio

        self.stimuli_type = "audio"
        self.source_link = "https://doi.org/10.1038/s41597-023-02752-5"
        super(GWilliams, self).__init__()

    def clean_recording(
        self,
        subject: int,
        task: int,
        session: int,
        notch_filter: bool = False,
        n_jobs: int = None,
    ) -> tuple[mne.io.Raw, pd.DataFrame, pd.DataFrame]:
        """Returns the clean recording containing MEG channels and the relevant events.

        Arguments:
            subject -- subject number (e.g. 0)
            task -- task number (e.g. 0)
            session -- session number (e.g. 0)
            notch_filter -- whether to apply notch filter to the raw data to remove powerline

        Returns:
            tuple of:
                raw -- mne.io.Raw containing the MEG data, notch filtered at 50, 100, 150, 200, 300, 400 Hz
                word_events -- DataFrame containing the word events. Columns are 'onset', 'duration', 'word'
                sound_events -- DataFrame containing the sound events. Columns are 'onset', 'sound', 'end'
                    onset is event marker in the brain data, start is the onset in the audio file
        """
        bids_path = self.recordings[subject][task][session]
        raw = mne_bids.read_raw_bids(bids_path, verbose=False)

        # Filter to only contain relevant channels
        raw = raw.pick(picks=self.channel_names, verbose=False)
        raw = raw.load_data(verbose=False)

        if notch_filter:
            # Determined by visual inspection of the data, exclude powerline noise
            raw = raw.notch_filter(
                freqs=[50, 100, 150, 200, 300, 400], verbose=False, n_jobs=n_jobs
            )

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
        sound_events["sound"] = sound_events["description"].apply(
            lambda x: json.loads(x.replace("'", '"'))["sound"].lower()
        )
        # So we only keep the path name in 'sound' column
        sound_events = sound_events[
            ~sound_events["sound"].str.contains("\.0")
        ].drop_duplicates(subset="onset", keep="first")
        sound_events.drop(
            ["description", "orig_time", "duration"], axis=1, inplace=True
        )

        # Add end time to each sound file
        for sound_file in sorted(sound_events["sound"].unique()):
            end_time = sound_events[sound_events["sound"] == sound_file]["onset"].iloc[
                -1
            ]
            sound_events.loc[sound_events["sound"] == sound_file, "end"] = end_time
        sound_events.drop_duplicates(subset="sound", keep="first", inplace=True)

        return raw, word_events, sound_events
