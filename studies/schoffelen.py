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
from studies.study import Study
import numpy as np

filterwarnings("ignore")


class Schoffelen(Study):

    def __init__(self, path: str = "data/schoffelen"):
        root_dir = os.path.join(os.getcwd(), path)
        assert os.path.exists(root_dir), f"{root_dir} does not exist"

        self.root_dir = root_dir
        self.cache_dir = os.path.join(os.getcwd(), "cache", "schoffelen")

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

        self.audio_cache = {}
        # Search for all audio files in the root directory and its subdirectories
        wav_files = list(pathlib.Path(self.root_dir).rglob("*.wav"))
        for wav_file in wav_files:
            audio, _ = librosa.load(wav_file, sr=16000)
            audio = audio.astype(np.float32)
            self.audio_cache[wav_file.name] = audio

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
            lambda x: f"{self.tasks[task][1:]}_{x[0]}.wav"
        )
        sound_events = sound_events.drop(columns=["value"])

        return raw, word_events, sound_events
