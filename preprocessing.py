import mne, os, glob
import matplotlib.pyplot as plt
from mne.channels import make_standard_montage
from mne.datasets import eegbci

def fetch_events(data_filtered, tmin=-1., tmax=4.):
    event_ids = dict(T1=0, T2=1)
    events, _ = mne.events_from_annotations(data_filtered, event_id=event_ids)
    picks = mne.pick_types(data_filtered.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    epochs = mne.Epochs(data_filtered, events, event_ids, tmin, tmax, proj=True,
                        picks=picks, baseline=None, preload=True)
    labels = epochs.events[:, -1]
    return labels, epochs


def filter_data(raw, plot, montage=make_standard_montage('standard_1020')):
    data_filter = raw.copy()
    data_filter.set_montage(montage)
    data_filter.filter(7, 30, fir_design='firwin', skip_by_annotation='edge')
    if plot:
        print("plotting after modification")
        mne.viz.plot_raw(data_filter, scalings={"eeg": 75e-6}, show=False)
        plt.show()
    return data_filter


def prepare_data(raw, plot, montage=make_standard_montage('standard_1020')):
    raw.rename_channels(lambda x: x.strip('.'))
    eegbci.standardize(raw)
    raw.set_montage(montage)

    # plot
    if plot:
        montage = raw.get_montage()
        print("plotting before modification")
        montage.plot(show=False)
        mne.viz.plot_raw(raw, scalings={"eeg": 75e-6}, show=False)
        plt.show()
    return raw

def fetch_data(subject_folder_path, sfreq=None):
    dataset = []
    subject = []
    files = sorted(glob.glob(os.path.join(subject_folder_path, f"{subject_folder_path[-4:]}R[0-9][0-9].edf")))
    for file in files:
        if file.endswith(".edf"):
            subject_data = mne.io.read_raw_edf(os.path.join(subject_folder_path, file), preload=True)
            if sfreq is None:
                sfreq = subject_data.info["sfreq"]
            if subject_data.info["sfreq"] == sfreq:
                subject.append(subject_data)
            else:
                break
    dataset.append(mne.concatenate_raws(subject))
    raw = mne.io.concatenate_raws(dataset)
    return raw

def process(subject_folder_path, plot=False):
    raw = fetch_data(subject_folder_path=subject_folder_path)
    raw = prepare_data(raw, plot)
    fildered_data = filter_data(raw, plot)
    return fetch_events(fildered_data)
