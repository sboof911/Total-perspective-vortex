import pandas as pd
import numpy as np
import mne

subject = 1
file = 3

classes = {
    0:"eyes open", # T0 file 001
    1:"eyes closed", # T0 file 002
    2:"left fist", # T1 files 3, 4, 7, 8, 11, 12
    3:"both fists", # T1 files 5, 6, 9, 10, 13, 14
    4:"right fist", # T2 files 3, 4, 7, 8, 11, 12
    5:"both feet" # T2 files 5, 6, 9, 10, 13, 14
}
fileName = f'/nfs/homes/amaach/sgoinfre/amaach/Downloads/google/eeg-motor-movementimagery-dataset-1.0.0/files/S{subject:03d}/S{subject:03d}R{file:02d}.edf'

reader = mne.io.read_raw_edf(fileName,preload=True)
annotations = reader.annotations
codes = annotations.description

df = pd.DataFrame(reader.get_data().T, columns=[channel.replace(".","") for channel in reader.ch_names])
df = df[~(df == 0).all(axis=1)]
timeArray = np.array([round(x,5) for x in np.arange(0,len(df)/160,(1/160))])
codeArray = []
counter = 0
for timeVal in timeArray:
    if timeVal in annotations.onset:
        counter += 1
    codeArray.append(codes[counter-1])

df["target"] = np.array(codeArray).T
print(df)
