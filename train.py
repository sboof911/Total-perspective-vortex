from preprocessing import preprocess, glob, os
from mne.decoding import CSP
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import pickle

def get_classifiers():
    return [
        OneVsRestClassifier(MLPClassifier(max_iter=2000)),
        OneVsRestClassifier(LogisticRegression()),
        OneVsRestClassifier(SVC(kernel='linear', probability=True)),
        OneVsRestClassifier(RandomForestClassifier(random_state=42)),
        OneVsRestClassifier(GradientBoostingClassifier()),
        OneVsRestClassifier(LogisticRegression(penalty='l1', solver='liblinear'))
    ]

class train:
    def __init__(self, clf=None) -> None:
        csp = CSP()
        if clf is None:
            clf = get_classifiers()[0]

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

def training_All_Data(preprocessmodule : preprocess, path):
    classifiers = get_classifiers()
    train_modules = []
    print("Training on all classifiers...")
    subjects_path = sorted(glob.glob(os.path.join(path, "S[1-9][0-9][0-9]"))) if not isinstance(path, list) else path
    experiments_accuracies = []
    for key, classifier in enumerate(classifiers):
        train_module = train(classifier)
        accuracies = []
        if len(subjects_path) == 0:
            raise Exception(f"No subjects found in {path}")
        for subject_path in subjects_path:
            labels, epochs_data_train = preprocessmodule.process(subject_path)
            accuracy, _ = train_module.fit(epochs_data_train, labels)
            accuracies.append(accuracy)
            print(f"experiment {key+1}/{len(classifiers)}: subject {subject_path[-4:]}: accuracy = {accuracy:.4f}")
        mean_accuracy = np.mean(accuracies)
        print(f"Mean accuracy of experiment {key+1}: {mean_accuracy:.4f}")
        experiments_accuracies.append(mean_accuracy)
        train_modules.append(train_modules)

    mean_accuracy = np.mean(experiments_accuracies)
    print("Mean accuracy of the six different experiments for all 109 subjects:")
    for key, accuracy in enumerate(experiments_accuracies):
        print(f"experiment {key+1}/{len(experiments_accuracies)}:{" "*10} accuracy = {accuracy:.4f}")
    print("", f"Mean accuracy of 6 experiments: {mean_accuracy}", sep='\n')
    best_module : train = train_modules[experiments_accuracies.index(max(experiments_accuracies))]
    return best_module

def training(preprocessmodule : preprocess, path, multiple_classifiers=False):
    if not multiple_classifiers:
        train_module = train()
        scores = []
        for subject_path in path:
            labels, epochs_data_train = preprocessmodule.process(subject_path)
            _, score = train_module.fit(epochs_data_train, labels)
            scores.append(score)
        return np.mean(scores, axis=0) if len(scores) > 1 else scores[0]
    else:
        best_module = training_All_Data(preprocessmodule, path)

        with open('model.pkl', 'wb') as file:
            pickle.dump(best_module.pipeline, file)
        return None