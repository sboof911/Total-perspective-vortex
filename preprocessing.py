import mne, os, glob
import matplotlib.pyplot as plt
from mne.channels import make_standard_montage
from mne.datasets import eegbci
import numpy as np

class preprocess:
    def __init__(self, plot=False):
        self._classes = [
            "Rest",
            "Open left fist",
            "Open right fist",
            "Imagine left fist",
            "Imagine right fist",
            "Open both fists",
            "Open both feet",
            "Imagine both fists",
            "Imagine both feet"
        ]
        self._dict = {classe:key for key, classe in enumerate(self._classes)}
        self._plot = plot
        mne.set_log_level('WARNING')

    def get_classes_name(self, filepath : str):
        dot = filepath.find('.')
        file_num = int(filepath[dot-2:dot])
        if file_num <= 2:
            return dict(T0=self._classes[0])
        else:
            count = file_num - 3
            return dict(T0=self._classes[0],
                        T1=self._classes[1+2*(count%4)],
                        T2=self._classes[2+2*(count%4)])

    def set_dict(self, dict_value):
        if not isinstance(dict_value, dict):
            raise Exception(f"dict_value must be dict. Value entered: {type(dict_value)}")
        self._dict = dict_value

    def change_event_names(self, task_data, file):
        classes_name = self.get_classes_name(file)
        annotations = task_data.annotations
        new_descriptions = [classes_name[description] for description in annotations.description]
        new_annotations = mne.Annotations(onset=annotations.onset,
                                  duration=annotations.duration,
                                  description=new_descriptions,
                                  orig_time=annotations.orig_time,
                                  ch_names=annotations.ch_names)

        # Assign the new annotations to the raw object
        task_data.set_annotations(new_annotations)
        return task_data

    def fetch_events(self, data_filtered, tmin=-1., tmax=4.):
        event_ids = self._dict
        events, _ = mne.events_from_annotations(data_filtered, event_id=event_ids)
        picks = mne.pick_types(data_filtered.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
        epochs = mne.Epochs(data_filtered, events, event_ids, tmin, tmax, proj=True,
                            picks=picks, baseline=None, preload=True)
        labels = epochs.events[:, -1]
        return labels, epochs.get_data()

    def filter_data(self, raw, montage=make_standard_montage('standard_1020')):
        data_filter = raw.copy()
        data_filter.set_montage(montage)
        data_filter.filter(7, 30, fir_design='firwin', skip_by_annotation='edge')

        if self._plot:
            print("plotting after modification")
            mne.viz.plot_raw(data_filter, scalings={"eeg": 75e-6}, show=False)
            plt.show()
        return data_filter

    def prepare_data(self, raw, montage=make_standard_montage('standard_1020')):
        raw.rename_channels(lambda x: x.strip('.'))
        eegbci.standardize(raw)
        raw.set_montage(montage)

        # plot
        if self._plot:
            montage = raw.get_montage()
            print("plotting before modification")
            montage.plot(show=False)
            mne.viz.plot_raw(raw, scalings={"eeg": 75e-6}, show=False)
            plt.show()
        return raw

    def fetch_data(self, subject_folder_path, sfreq=None):
        subject = []
        if not os.path.isfile(subject_folder_path):
            files = sorted(glob.glob(os.path.join(subject_folder_path, f"{subject_folder_path[-4:]}R[0-9][0-9].edf")))
            if len(files) == 0:
                raise Exception("No tasks found!")
        else:
            files = [subject_folder_path]

        for file in files:
            task_data = mne.io.read_raw_edf(file, preload=True)
            task_data = self.change_event_names(task_data, file)
            if sfreq is None:
                sfreq = task_data.info["sfreq"]
            if task_data.info["sfreq"] == sfreq:
                subject.append(task_data)
            else:
                raise Exception("A task has different samples frequence number!")
        raw = mne.io.concatenate_raws(subject)
        return raw

    def process(self, subject_folder_path):
        raw = self.fetch_data(subject_folder_path)
        raw = self.prepare_data(raw)
        fildered_data = self.filter_data(raw)
        return self.fetch_events(fildered_data)
