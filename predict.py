from preprocessing import preprocess
import pickle
import numpy as np

def predict(preprocessmodule : preprocess, args, folder_path):
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    file_path = f"{folder_path}/S{args.subject_num:03}/S{args.subject_num:03}R{args.task_num:02}.edf"
    labels, epochs = preprocessmodule.process(file_path)
    epochs = epochs.get_data()

    print("X shape= ", epochs.shape, "y shape= ", labels.shape)

    scores = []
    for n in range(epochs.shape[0]):
        pred = model.predict(epochs[n:n + 1, :, :])
        print("pred= ", pred, "truth= ", labels[n:n + 1])
        scores.append(1 - np.abs(pred[0] - labels[n:n + 1][0]))
    print("Mean acc= ", np.mean(scores))
