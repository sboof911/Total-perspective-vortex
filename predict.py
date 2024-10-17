from preprocessing import preprocess
import pickle
import numpy as np

def predict(preprocessmodule : preprocess, data_num, folder_path):
    with open('./model.pkl', 'rb') as file:
        model = pickle.load(file)
    file_path = f"{folder_path}/S{data_num[0]:03}/S{data_num[0]:03}R{data_num[1]:02}.edf"
    if data_num[1] in [1, 2]:
        preprocessmodule.set_dict(dict(T0=0))
    labels, epochs = preprocessmodule.process(file_path)
    epochs = epochs.get_data()

    print("X shape= ", epochs.shape, "y shape= ", labels.shape)

    scores = []
    for n in range(epochs.shape[0]):
        pred = model.predict(epochs[n:n + 1, :, :])
        print("pred= ", pred, "truth= ", labels[n:n + 1])
        scores.append(1 - np.abs(pred[0] - labels[n:n + 1][0]))
    print("Mean acc= ", np.mean(scores))
