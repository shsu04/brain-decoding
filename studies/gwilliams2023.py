from itertools import product
import json
import os
import copy
import pathlib
import mne
import mne_bids
import numpy as np
import pandas as pd
from warnings import filterwarnings
from .study import Study, Recording
from .stimuli import Stimuli
from .download import download_osf

filterwarnings("ignore")


class Gwilliams2023(Study):

    paper_link = "https://doi.org/10.1038/s41597-023-02752-5"

    def __init__(
        self,
        batch_type: str,
        path: str = "data/gwilliams2023",
        cache_enabled: bool = True,
        max_cache_size: int = 100,
        cache_name: str = "cache",
    ):
        root_dir = os.path.join(os.getcwd(), path)

        # Download
        if not os.path.exists(root_dir):
            download_bool = input(
                f"{root_dir} not found. Do you want to download it? [y/n]: "
            )
            if download_bool == "y":
                self.download(root_dir)

        assert os.path.exists(root_dir), f"{root_dir} does not exist"

        self.root_dir = root_dir
        self.cache_dir = os.path.join(os.getcwd(), cache_name, "gwilliams2023")

        # Create cache directory
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        self.subjects_info = pd.read_csv(
            os.path.join(self.root_dir, "participants.tsv"), sep="\t"
        )

        self.subjects = (
            self.subjects_info["participant_id"].str.split("-").str[1].tolist()
        )
        self.sessions = [str(i) for i in range(2)]
        self.tasks = [str(i) for i in range(4)]  # story_uid

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

        self.batch_type = batch_type
        print(f"Loading {self.__class__.__name__} with batch type {self.batch_type}")

        # Load the thread-safe stimuli manager
        self.wav_paths = list(pathlib.Path(self.root_dir).rglob("*.wav"))
        self.stimuli = [
            Stimuli(
                root_path=self.root_dir,
                names=[f"stimuli/audio/{wav_file.name}" for wav_file in self.wav_paths],
                cache_enabled=cache_enabled,
                max_cache_size=max_cache_size,
            )
            for _ in range(len(self.tasks))
        ]
        self.create_recordings()

        super(Gwilliams2023, self).__init__()

    def create_recordings(self):
        # Recordings is a 3D array, where the first dimension is the subject,
        # the second dimension is the session, and the third dimension is the task.
        self.recordings = [
            [[] for _ in range(len(self.sessions))] for i in range(len(self.subjects))
        ]
        # Create the recordings
        for subject, session, task in product(
            range(len(self.subjects)), range(len(self.sessions)), range(len(self.tasks))
        ):

            bids_path = mne_bids.BIDSPath(
                subject=self.subjects[subject],
                session=self.sessions[session],
                task=self.tasks[task],
                datatype="meg",
                root=self.root_dir,
            )

            # Not all subjects did 2 sessions, but all sessions had 4 tasks
            if not bids_path.fpath.exists():
                continue

            self.recordings[subject][session].append(
                Gwilliams2023Recording(
                    bids_path=bids_path,
                    cache_path=os.path.join(
                        self.cache_dir, f"sub_{subject}_ses_{session}_task_{task}"
                    ),
                    study_name="Gwilliams2023",
                    subject_id=self.subjects[subject],
                    session_id=self.sessions[session],
                    task_id=self.tasks[task],
                    channel_names=copy.copy(self.channel_names),
                    stimuli=self.stimuli[task],
                    power_line_freq=50,
                    type=self.batch_type,
                )
            )

    def download(self, root_dir: str):
        """Downloads the data from the repository."""
        # Download the data
        print(f"Downloading Gwilliams2023 data to {root_dir}...")
        download_osf(root_dir, project_ids=["ag3kj", "hqvm3", "u5327", "dr4wy"])
        print("Downloaded Gwilliams2023.")
        return


class Gwilliams2023Recording(Recording):
    def __init__(
        self,
        bids_path: str,
        cache_path: str,
        study_name: str,
        subject_id: str,
        session_id: str,
        task_id: str,
        channel_names: list[str],
        stimuli: Stimuli,
        power_line_freq=50,
        type: str = "audio",
    ):
        super(Gwilliams2023Recording, self).__init__(
            bids_path=bids_path,
            cache_path=cache_path,
            study_name=study_name,
            subject_id=subject_id,
            session_id=session_id,
            task_id=task_id,
            channel_names=channel_names,
            stimuli=stimuli,
            power_line_freq=power_line_freq,
            type=type,
        )

    def load_raw(self, load_data: bool) -> mne.io.Raw:
        """Loads the raw data fwith only the pre-defined relevant channels."""
        raw = mne_bids.read_raw_bids(self.bids_path, verbose=False)
        # Filter to only contain relevant channels
        raw = raw.pick(picks=self.channel_names, verbose=False)
        self.info = raw.info  # for later access

        # Load the data only if needed for efficiency
        if load_data:
            raw.load_data(verbose=False)

        return raw

    def load_events(self, raw: mne.io.Raw, options: str) -> dict[str, pd.DataFrame]:
        """Loads the events from the raw data.

        Arguments:
            options -- either 'word', 'sound', or 'both'

        Returns:
            dict[str, pd.DataFrame] -- dictionary containing the word and sound events

            word_events -- DataFrame containing the word events. Columns are 'onset', 'duration', 'word'
            sound_events -- DataFrame containing the sound events. Columns are 'onset', 'sound', 'end'
                onset is event marker in the brain data, start is the onset in the audio file
        """
        assert options in [
            "word",
            "sound",
            "both",
        ], "Options must be either 'word' or 'sound'."

        # Contains all events
        annotations_df = pd.DataFrame(raw.annotations)
        word_events, sound_events, results = None, None, {}

        # Word events
        if options == "word" or options == "both":
            word_events = copy.deepcopy(annotations_df)

            # Filter out events description to only contain pronounced word
            word_events = word_events[
                word_events["description"].str.contains("'kind': 'word'")
                & word_events["description"].str.contains("'pronounced': 1.0")
            ].reset_index(drop=True)

            word_events["word"] = word_events["description"].apply(
                lambda x: json.loads(x.replace("'", '"'))["word"]
            )
            word_events.drop(["description", "orig_time"], axis=1, inplace=True)
            results["word"] = word_events

        # Sound events
        if options == "sound" or options == "both":
            sound_events = copy.deepcopy(annotations_df)

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
                end_time = sound_events[sound_events["sound"] == sound_file][
                    "onset"
                ].iloc[-1]
                sound_events.loc[sound_events["sound"] == sound_file, "end"] = end_time

            sound_events.drop_duplicates(subset="sound", keep="first", inplace=True)
            results["sound"] = sound_events

        del annotations_df
        return results

    # Thread-safe between recording instances
    def load_stimuli(self, names: list[str], options: str = None) -> np.ndarray:
        return self.stimuli.load_audio(names=names)
