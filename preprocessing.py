import argparse, mne, os, glob
import pandas as pd
import numpy as np

class dataset:
    def __init__(self, folderpath, tosavepath) -> None:
        if not os.path.exists(folderpath):
            raise Exception(f"{folderpath} not found!")
        if not os.path.exists(tosavepath):
            os.makedirs(tosavepath)
        self._datafolderPath = folderpath
        self._savedata_folderPath = tosavepath
        self._subjects_paths = sorted(glob.glob(os.path.join(folderpath, "S[0-9][0-9][0-9]")))
        if len(self._subjects_paths) == 0:
            raise Exception(f"No subjects found in {folderpath}")

    def get_filter_data(self, task_path):
        raw = mne.io.read_raw_edf(task_path,preload=True, verbose='WARNING')
        raw.filter(1., 40., fir_design='firwin')
        # Removing eye blinks
        ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
        ica.fit(raw)
        eog_indices, _ = ica.find_bads_eog(raw)
        ica.exclude = eog_indices
        raw = ica.apply(raw)

        return raw

    def get_dataFrame(self, mne_data):
        annotations = mne_data.annotations
        codes = annotations.description
        df = pd.DataFrame(mne_data.get_data().T, columns=[channel.replace(".","") for channel in mne_data.ch_names])
        df = df[~(df == 0).all(axis=1)]
        timeArray = np.array([round(x,5) for x in np.arange(0,len(df)/160,(1/160))])
        codeArray = []
        counter = 0
        for timeVal in timeArray:
            if timeVal in annotations.onset:
                counter += 1
            codeArray.append(codes[counter-1])

        df["target"] = np.array(codeArray).T
        return df

    def fuse_DataFrame(self, global_df : pd.DataFrame, next_df : pd.DataFrame) -> pd.DataFrame:
        if global_df is None:
            return next_df
        if not all(global_df.columns == next_df.columns):
            raise Exception("columns are different!")
        return pd.concat([global_df, next_df], axis=0)

    def preprocess_tasks(self, subject_path):
        tasks_path = sorted(glob.glob(os.path.join(subject_path, f"{subject_path[-4:]}R[0-9][0-9].edf")))
        DataFrame = None
        classes = {
            0: "Rest",
            1: "Real Left Fist Motion",
            2: "Real Right Fist Motion",
            3: "Imagined Left Fist Motion",
            4: "Imagined Right Fist Motion",
            5: "Real Both Fists Motion",
            6: "Real Both Feet Motion",
            7: "Imagined Both Fists Motion",
            8: "Imagined Both Feet Motion"
        }
        mapping = {
            0: {'T0': classes[0], 'T1': classes[1], 'T2': classes[2]},
            1: {'T0': classes[0], 'T1': classes[3], 'T2': classes[4]},
            2: {'T0': classes[0], 'T1': classes[5], 'T2': classes[6]},
            3: {'T0': classes[0], 'T1': classes[7], 'T2': classes[8]}
        }
        count = -2
        for task_path in tasks_path:
            if not os.path.exists(task_path):
                raise Exception(f"{task_path} not found!")
            filder_mne_data = self.get_filter_data(task_path)
            # change target to the named files
            df = self.get_dataFrame(filder_mne_data)
            if count >= 0:
                df['target'] = df['target'].replace(mapping[count % 4])
            else:
                df['target'] = df['target'].replace({'T0':classes[0]})
            count = count + 1
            DataFrame = self.fuse_DataFrame(DataFrame, df)
            

        DataFrame.to_csv("/nfs/homes/amaach/Desktop/Total-perspective-vortex/datatest.csv", index=False) # to change

    def preprocess(self):
        for subject_path in self._subjects_paths:
            print(f"subject preprocess: {subject_path}")
            self.preprocess_tasks(subject_path)
            break

    def test_plot(self):
        pass

def get_args():
    parser = argparse.ArgumentParser(description="Preprocess a EEG signal Data.")

    parser.add_argument("--data_train", type=str, required=True, help="Folder path to the datatrain.")
    parser.add_argument("--data_save", type=str, required=True, help="Folder path to where save the data.")
    parser.add_argument("--subject_number", type=int, default=-1, help="Folder path to where save the data.")

    return parser.parse_args()

if __name__ == "__main__":
    try:
        args = get_args()
        data = dataset(args.data_train, args.data_save)
        data.preprocess()
    except Exception as e:
        print(e)
