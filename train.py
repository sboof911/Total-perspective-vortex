from preprocessing import preprocess, glob, os
from mne.decoding import CSP
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
import numpy as np

class train:
    def __init__(self) -> None:
        csp = CSP()
        clf = OneVsRestClassifier(LogisticRegression(max_iter=100))

        self._pipeline = Pipeline([
            ('csp', csp),
            ('classifier', clf)
        ])
        self._cv = StratifiedKFold(n_splits=10)

    def fit(self, epochs_data_train, labels):
        # Evaluate model performance using cross-validation on the training data
        X_train, X_test, y_train, y_test = train_test_split(epochs_data_train, labels, test_size=0.2, random_state=42)
        scores = cross_val_score(self._pipeline, X_train, y_train, cv=self._cv, scoring='accuracy')
        self._pipeline.fit(X_train, y_train)

        # Test Data
        y_pred = self._pipeline.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        accuracy = report['accuracy']
        return accuracy, scores

    @property
    def pipeline(self):
        return self._pipeline

def training(preprocessmodule : preprocess, train_module : train, path, multiple_subjects=False):
    def launch(subject_path):
        labels, epochs = preprocessmodule.process(subject_path)
        epochs_data_train = epochs.get_data()
        return train_module.fit(epochs_data_train, labels)

    if not multiple_subjects:
        return launch(path)
    else:
        print("Training on all subjects...")
        subjects_path = sorted(glob.glob(os.path.join(path, "S[0-9][0-9][0-9]")))
        accuracies = []
        if len(subjects_path) == 0:
            raise Exception(f"No subjects found in {path}")
        for key, subject_path in enumerate(subjects_path):
            accuracy, scores = launch(subject_path)
            accuracies.append(accuracy)
            print(f"experiment 0: subject {key:03}: accuracy = {accuracy}")
        mean_accuracy = np.mean(accuracies)
        print(f"Mean accuracy of experiment 0: {mean_accuracy}")
