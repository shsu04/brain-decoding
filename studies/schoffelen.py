# from itertools import product
# import json
# import os
# import pandas as pd
# import mne_bids
# import copy
# import mne
# from warnings import filterwarnings

# from studies.study import Study


# class Schoffelen(Study):

#     def __init__(self, path: str = "data/schoffelen"):
#         root_dir = os.path.join(os.getcwd(), path)
#         assert os.path.exists(root_dir), f"{root_dir} does not exist"

#         self.root_dir = root_dir
#         self.subjects_info = pd.read_csv(
#             os.path.join(self.root_dir, "participants.tsv"), sep="\t"
#         )

#         # Subject IDs list in string form, e.g. 000, 001, etc.
#         self.subjects_list = (
#             self.subjects_info["participant_id"].str.split("-").str[1].tolist()
#         )

#         self.sessions = ["compr"]
#         self.tasks = [
#             "001",
#             "002",
#             "003",
#             "004",
#             "005",
#             "006",
#             "007",
#             "008",
#             "009",
#             "010",
#         ]

#         self.recordings = {f"{i}": [[] for _ in range(10)] for i in self.subjects_list}

#         for subject, session, task in product(
#             self.subjects_list, self.sessions, self.tasks
#         ):
#             # Task and session are swapped in the BIDSPath, since in
#             # this study the naming convention is swapped

#             bids_path = mne_bids.BIDSPath(
#                 subject=subject,
#                 session=task,
#                 task=session,
#                 root=self.root_dir,
#             )

#             if not bids_path.fpath.exists():
#                 raise FileNotFoundError(f"{bids_path.fpath} does not exist")

#             self.recordings[subject][int(task) - 1].append(bids_path)

#         # The only valid channel types in this study
#         self.types = [
#             "MLC",
#             "MLF",
#             "MLO",
#             "MLP",
#             "MLT",
#             "MRC",
#             "MRF",
#             "MRO",
#             "MRP",
#             "MRT",
#             "MZC",
#             "MZF",
#             "MZO",
#             "MZP",
#         ]

#         self.stimuli_type = "audio"
#         self.source_link = "https://www.nature.com/articles/s41597-022-01382-7"
#         super(Schoffelen, self).__init__()

#     def clean_recording(
#         self, subject: str, task: int, session, n_jobs: int = None,
#     ) -> tuple[pd.DataFrame, pd.DataFrame]:
#         """Returns the clean recording containing MEG channels and the relevant events.

#         Arguments:
#             subject -- subject ID (e.g. '01')
#             task -- task number (e.g. 0)
#             session -- session number (e.g. 0)

#         Returns:
#             tuple of:
#                 raw -- mne.io.Raw containing the MEG data, notch filtered at 50, 100, 150, 200, 300, 400 Hz
#                 word_events -- DataFrame containing the word events. Columns are 'onset', 'duration', 'word'
#                 sound_events -- DataFrame containing the sound events. Columns are 'onset', 'sound', 'start'
#                     onset is event marker in the brain data, start is the onset in the audio file
#         """
#         # Schoffelen has only one session
#         bids_path = self.recordings[subject][task][0]
#         raw = mne.read_raw(bids_path, verbose=False)
#         # plot psd
#         raw.compute_psd().plot()

#         if not hasattr(self, "old_sample_rate"):
#             self.old_sample_rate = raw.info["sfreq"]
#         if not hasattr(self, "info"):
#             self.info = raw.info

#         if not hasattr(self, "channel_names"):
#             self.channel_names = [
#                 channel
#                 for channel in raw.ch_names
#                 if any([type in channel for type in self.types])
#             ]

#         raw = raw.pick(picks=self.channel_names, verbose=False).load_data(verbose=False)
# # Determined by visual inspection of the data
# raw = raw.notch_filter(
#     freqs=[50, 100, 150, 200, 300, 400], verbose=False, n_jobs=n_jobs
# )
# # The time columns
# time = pd.Series(raw.times, name="time")
# # Filter to only contain relevant channels, reshape df from [Ch, Time] to [Time, Ch]
# raw_df = pd.concat(
#     [time, pd.DataFrame(raw.get_data(picks=self.channel_names)).T], axis=1
# )

# events_path = bids_path.copy().update(suffix="events", extension=".tsv")

# # We do this since this study does not have complete events from the mne.Raw object
# annotations_df = pd.read_csv(
#     str(events_path.directory) + "/meg/" + events_path.basename, sep="\t"
# )
# # Filter out phonemes and other irrelevant annotations
# annotations_df = annotations_df[
#     annotations_df["type"].str.contains("word_onset")
# ].drop(columns=["sample", "type"])

# # Rename to match GWilliams
# annotations_df.columns = ["onset", "duration", "word"]

# annotations_df["word"] = annotations_df["word"].apply(lambda x: x.lower())
# # 'sp' is recurrent, but author did not provide explaination
# annotations_df = annotations_df[annotations_df["word"] != "sp"].reset_index(
#     drop=True
# )

# return raw_df, annotations_df
