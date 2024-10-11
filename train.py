from preprocessing import preprocess, glob, os
from mne.decoding import CSP
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import pickle

class train:
    def __init__(self, clf=None) -> None:
        csp = CSP()
        if clf is None:
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

def launch(preprocessmodule : preprocess, subject_path, train_module):
        labels, epochs = preprocessmodule.process(subject_path)
        epochs_data_train = epochs.get_data()
        return train_module.fit(epochs_data_train, labels)

def training_All_Data(preprocessmodule : preprocess, path):
    classifiers = [
        OneVsRestClassifier(LogisticRegression(max_iter=100)),
        OneVsRestClassifier(SVC(kernel='linear', probability=True)),
        OneVsRestClassifier(RandomForestClassifier(n_estimators=100)),
        OneVsRestClassifier(GradientBoostingClassifier(n_estimators=100)),
        OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5)),
        OneVsRestClassifier(MLPClassifier(max_iter=100))
    ]
    train_modules = []
    print("Training on all subjects...")
    subjects_path = sorted(glob.glob(os.path.join(path, "S[0-9][0-9][0-9]")))
    experiments_accuracies = []
    for key, classifier in enumerate(classifiers):
        train_module = train(classifier)
        accuracies = []
        if len(subjects_path) == 0:
            raise Exception(f"No subjects found in {path}")
        for subject_path in subjects_path:
            accuracy, _ = launch(preprocessmodule, subject_path, train_module)
            accuracies.append(accuracy)
            print(f"experiment {key}/{len(classifiers)}: subject {subject_path[-4:]}: accuracy = {accuracy:.4f}")
        mean_accuracy = np.mean(accuracies)
        print(f"Mean accuracy of experiment {key}: {mean_accuracy:.4f}")
        experiments_accuracies.append(mean_accuracy)
        train_modules.append(train_modules)

    mean_accuracy = np.mean(experiments_accuracies)
    print("Mean accuracy of the six different experiments for all 109 subjects:")
    for key, accuracy in enumerate(experiments_accuracies):
        print(f"experiment {key}:           accuracy = {accuracy:.4f}")
    print("", f"Mean accuracy of 6 experiments: {mean_accuracy}", sep='\n')
    best_module : train = train_modules[experiments_accuracies.index(max(experiments_accuracies))]
    return best_module

def training(preprocessmodule : preprocess, path, multiple_subjects=False):
    if not multiple_subjects:
        train_module = train()
        return launch(preprocessmodule, path, train_module)
    else:
        best_module, mean_accuracy = training_All_Data(preprocessmodule, path)

        with open('model.pkl', 'wb') as file:
            pickle.dump(best_module.pipeline, file)
        return mean_accuracy
