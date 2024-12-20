from asyncio import tasks
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

filterwarnings("ignore")


class Schoffelen2022(Study):
    def __init__(
        self,
        batch_type: str,
        path: str = "data/schoffelen2022",
        cache_enabled: bool = True,
        max_cache_size: int = 100,
        cache_name: str = "cache",
    ):
        root_dir = os.path.join(os.getcwd(), path)
        assert os.path.exists(root_dir), f"{root_dir} does not exist"

        self.root_dir = root_dir
        self.cache_dir = os.path.join(os.getcwd(), cache_name, "schoffelen2022")

        # Create cache directory
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        self.subjects_info = pd.read_csv(
            os.path.join(self.root_dir, "participants.tsv"), sep="\t"
        )

        # Subject IDs list in string form, e.g. 000, 001, etc.
        self.subjects = (
            self.subjects_info["participant_id"].str.split("-").str[1].tolist()
        )

        self.sessions = [
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

        self.tasks = ["compr"]  # Not taking empty room

        # The only valid channel types in this study
        self.channel_names = [
            "MLC12-4304",
            "MLC13-4304",
            "MLC14-4304",
            "MLC15-4304",
            "MLC16-4304",
            "MLC17-4304",
            "MLC21-4304",
            "MLC22-4304",
            "MLC23-4304",
            "MLC24-4304",
            "MLC25-4304",
            "MLC31-4304",
            "MLC41-4304",
            "MLC42-4304",
            "MLC51-4304",
            "MLC52-4304",
            "MLC53-4304",
            "MLC54-4304",
            "MLC55-4304",
            "MLC62-4304",
            "MLC63-4304",
            "MLF11-4304",
            "MLF12-4304",
            "MLF13-4304",
            "MLF14-4304",
            "MLF21-4304",
            "MLF22-4304",
            "MLF23-4304",
            "MLF24-4304",
            "MLF25-4304",
            "MLF31-4304",
            "MLF32-4304",
            "MLF33-4304",
            "MLF34-4304",
            "MLF35-4304",
            "MLF41-4304",
            "MLF42-4304",
            "MLF43-4304",
            "MLF44-4304",
            "MLF45-4304",
            "MLF46-4304",
            "MLF51-4304",
            "MLF52-4304",
            "MLF53-4304",
            "MLF54-4304",
            "MLF55-4304",
            "MLF56-4304",
            "MLF61-4304",
            "MLF62-4304",
            "MLF63-4304",
            "MLF64-4304",
            "MLF65-4304",
            "MLF66-4304",
            "MLF67-4304",
            "MLO11-4304",
            "MLO12-4304",
            "MLO13-4304",
            "MLO14-4304",
            "MLO21-4304",
            "MLO22-4304",
            "MLO23-4304",
            "MLO24-4304",
            "MLO31-4304",
            "MLO32-4304",
            "MLO34-4304",
            "MLO41-4304",
            "MLO42-4304",
            "MLO43-4304",
            "MLO44-4304",
            "MLO51-4304",
            "MLO52-4304",
            "MLO53-4304",
            "MLP11-4304",
            "MLP12-4304",
            "MLP21-4304",
            "MLP22-4304",
            "MLP23-4304",
            "MLP31-4304",
            "MLP32-4304",
            "MLP33-4304",
            "MLP34-4304",
            "MLP35-4304",
            "MLP41-4304",
            "MLP42-4304",
            "MLP43-4304",
            "MLP44-4304",
            "MLP45-4304",
            "MLP51-4304",
            "MLP52-4304",
            "MLP53-4304",
            "MLP54-4304",
            "MLP55-4304",
            "MLP56-4304",
            "MLP57-4304",
            "MLT11-4304",
            "MLT12-4304",
            "MLT13-4304",
            "MLT14-4304",
            "MLT15-4304",
            "MLT16-4304",
            "MLT21-4304",
            "MLT22-4304",
            "MLT23-4304",
            "MLT24-4304",
            "MLT25-4304",
            "MLT26-4304",
            "MLT27-4304",
            "MLT31-4304",
            "MLT32-4304",
            "MLT33-4304",
            "MLT34-4304",
            "MLT35-4304",
            "MLT36-4304",
            "MLT37-4304",
            "MLT41-4304",
            "MLT42-4304",
            "MLT43-4304",
            "MLT44-4304",
            "MLT45-4304",
            "MLT46-4304",
            "MLT47-4304",
            "MLT51-4304",
            "MLT52-4304",
            "MLT53-4304",
            "MLT54-4304",
            "MLT55-4304",
            "MLT56-4304",
            "MLT57-4304",
            "MRC11-4304",
            "MRC12-4304",
            "MRC13-4304",
            "MRC14-4304",
            "MRC15-4304",
            "MRC16-4304",
            "MRC17-4304",
            "MRC21-4304",
            "MRC22-4304",
            "MRC23-4304",
            "MRC24-4304",
            "MRC25-4304",
            "MRC31-4304",
            "MRC32-4304",
            "MRC41-4304",
            "MRC42-4304",
            "MRC51-4304",
            "MRC52-4304",
            "MRC53-4304",
            "MRC54-4304",
            "MRC55-4304",
            "MRC61-4304",
            "MRC62-4304",
            "MRC63-4304",
            "MRF11-4304",
            "MRF12-4304",
            "MRF13-4304",
            "MRF14-4304",
            "MRF21-4304",
            "MRF22-4304",
            "MRF23-4304",
            "MRF24-4304",
            "MRF25-4304",
            "MRF31-4304",
            "MRF32-4304",
            "MRF33-4304",
            "MRF34-4304",
            "MRF35-4304",
            "MRF41-4304",
            "MRF42-4304",
            "MRF43-4304",
            "MRF44-4304",
            "MRF45-4304",
            "MRF46-4304",
            "MRF51-4304",
            "MRF52-4304",
            "MRF53-4304",
            "MRF54-4304",
            "MRF55-4304",
            "MRF56-4304",
            "MRF61-4304",
            "MRF62-4304",
            "MRF63-4304",
            "MRF64-4304",
            "MRF65-4304",
            "MRF67-4304",
            "MRO11-4304",
            "MRO12-4304",
            "MRO13-4304",
            "MRO14-4304",
            "MRO21-4304",
            "MRO22-4304",
            "MRO23-4304",
            "MRO24-4304",
            "MRO31-4304",
            "MRO32-4304",
            "MRO34-4304",
            "MRO41-4304",
            "MRO42-4304",
            "MRO43-4304",
            "MRO44-4304",
            "MRO51-4304",
            "MRO52-4304",
            "MRO53-4304",
            "MRP11-4304",
            "MRP12-4304",
            "MRP21-4304",
            "MRP22-4304",
            "MRP23-4304",
            "MRP31-4304",
            "MRP32-4304",
            "MRP33-4304",
            "MRP34-4304",
            "MRP35-4304",
            "MRP41-4304",
            "MRP42-4304",
            "MRP43-4304",
            "MRP44-4304",
            "MRP45-4304",
            "MRP51-4304",
            "MRP52-4304",
            "MRP53-4304",
            "MRP54-4304",
            "MRP55-4304",
            "MRP56-4304",
            "MRP57-4304",
            "MRT11-4304",
            "MRT12-4304",
            "MRT13-4304",
            "MRT14-4304",
            "MRT15-4304",
            "MRT16-4304",
            "MRT21-4304",
            "MRT22-4304",
            "MRT23-4304",
            "MRT24-4304",
            "MRT25-4304",
            "MRT26-4304",
            "MRT27-4304",
            "MRT31-4304",
            "MRT32-4304",
            "MRT33-4304",
            "MRT34-4304",
            "MRT35-4304",
            "MRT36-4304",
            "MRT37-4304",
            "MRT41-4304",
            "MRT42-4304",
            "MRT43-4304",
            "MRT44-4304",
            "MRT45-4304",
            "MRT46-4304",
            "MRT47-4304",
            "MRT51-4304",
            "MRT52-4304",
            "MRT53-4304",
            "MRT54-4304",
            "MRT55-4304",
            "MRT56-4304",
            "MRT57-4304",
            "MZC01-4304",
            "MZC02-4304",
            "MZC03-4304",
            "MZC04-4304",
            "MZF01-4304",
            "MZF02-4304",
            "MZF03-4304",
            "MZO01-4304",
            "MZO02-4304",
            "MZO03-4304",
            "MZP01-4304",
        ]

        self.source_link = "https://www.nature.com/articles/s41597-022-01382-7"
        self.batch_type = batch_type
        print(f"Loading {self.__class__.__name__} with batch type {self.batch_type}")

        # Load the thread-safe stimuli manager
        self.wav_paths = list(pathlib.Path(self.root_dir).rglob("*.wav"))
        self.stimuli = [
            Stimuli(
                root_path=self.root_dir + "/stimuli",
                names=[f"{wav_file.name}" for wav_file in self.wav_paths],
                cache_enabled=cache_enabled,
                max_cache_size=max_cache_size,
            )
            for _ in range(len(self.sessions))
        ]
        self.create_recordings()

        super(Schoffelen2022, self).__init__()

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

            # All recordings should be present
            if not bids_path.fpath.exists():
                raise FileNotFoundError(f"{bids_path.fpath} does not exist")

            self.recordings[subject][session].append(
                Schoffelen2022Recording(
                    bids_path=bids_path,
                    cache_path=os.path.join(
                        self.cache_dir, f"sub_{subject}_ses_{session}_task_{task}"
                    ),
                    study_name="Schoffelen2022",
                    subject_id=self.subjects[subject],
                    session_id=self.sessions[session],
                    task_id=self.tasks[task],
                    channel_names=copy.copy(self.channel_names),
                    stimuli=self.stimuli[session],
                    power_line_freq=50,
                    type=self.batch_type,
                )
            )


class Schoffelen2022Recording(Recording):
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
        super(Schoffelen2022Recording, self).__init__(
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

    def load_raw(
        self,
        load_data: bool = False,
    ) -> mne.io.Raw:
        """Loads the raw data with only the pre-defined relevant channels."""
        raw = mne.io.read_raw(self.bids_path, verbose=False)
        # Filter to only contain relevant channels
        raw = raw.pick(picks=self.channel_names, verbose=False)

        # Load the data only if needed for efficiency
        if load_data:
            raw.load_data(verbose=False)

        self.info = raw.info  # for later access
        return raw

    def load_events(
        self, raw: mne.io.Raw = None, options: str = None
    ) -> dict[str, pd.DataFrame]:
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

        # We do this since this study does not have complete events from the mne.Raw object
        events_path = self.bids_path.copy().update(suffix="events", extension=".tsv")
        annotations_df = pd.read_csv(
            str(events_path.directory) + "/" + events_path.basename, sep="\t"
        )

        word_events, sound_events, results = None, None, {}

        # Word events
        if options == "word" or options == "both":

            word_events = copy.deepcopy(annotations_df)
            # Filter out phonemes and other irrelevant annotations
            word_events = word_events[
                word_events["type"].str.contains("word_onset")
            ].drop(columns=["sample", "type"])

            # Rename to match Gwilliams
            word_events.columns = ["onset", "duration", "word"]

            word_events["word"] = word_events["word"].apply(lambda x: x.lower())
            # 'sp' is recurrent, but author did not provide explaination
            word_events = word_events[word_events["word"] != "sp"].reset_index(
                drop=True
            )
            results["word"] = word_events

        # Sound events
        if options == "sound" or options == "both":
            sound_events = copy.deepcopy(annotations_df)

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
            end_times = end_times.drop_duplicates(subset="type", keep="first")
            # So that columns are onset, sound, end
            sound_events["end"] = end_times["onset"].values
            # stimuli/{tasknum}_{num}.wav is the path to the audio file
            sound_events["sound"] = sound_events["value"].apply(
                lambda x: f"{self.session_id[1:]}_{x[0]}.wav"
            )
            sound_events = sound_events.drop(columns=["value"])
            results["sound"] = sound_events

        del annotations_df
        return results

    # Thread-safe between recording instances
    def load_stimuli(self, names: list[str], options: str = None) -> np.ndarray:
        return self.stimuli.load_audio(names=names)
