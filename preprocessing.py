import mne, os, glob
import matplotlib.pyplot as plt
from mne.channels import make_standard_montage
from mne.datasets import eegbci

class preprocess:
    def __init__(self, plot=False):
        self._dict = dict(T0=0, T1=1, T2=2)
        self._plot = plot
        mne.set_log_level('WARNING')

    def set_dict(self, dict_value):
        if not isinstance(dict_value, dict):
            raise Exception(f"dict_value must be dict. Value entered: {type(dict_value)}")
        self._dict = dict_value

    def fetch_events(self, data_filtered, tmin=-1., tmax=4.):
        event_ids = self._dict
        events, _ = mne.events_from_annotations(data_filtered, event_id=event_ids)
        picks = mne.pick_types(data_filtered.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
        epochs = mne.Epochs(data_filtered, events, event_ids, tmin, tmax, proj=True,
                            picks=picks, baseline=None, preload=True)
        labels = epochs.events[:, -1]
        return labels, epochs

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
            subject_data = mne.io.read_raw_edf(file, preload=True)
            if sfreq is None:
                sfreq = subject_data.info["sfreq"]
            if subject_data.info["sfreq"] == sfreq:
                subject.append(subject_data)
            else:
                raise Exception("A task has different samples frequence number!")
        raw = mne.io.concatenate_raws(subject)
        return raw

    def process(self, subject_folder_path):
        raw = self.fetch_data(subject_folder_path)
        raw = self.prepare_data(raw)
        fildered_data = self.filter_data(raw)
        return self.fetch_events(fildered_data)
