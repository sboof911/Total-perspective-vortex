from preprocessing import preprocess, glob
from mne.decoding import CSP
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier

class train:
    def __init__(self) -> None:
        csp = CSP(cov_est='epoch')
        clf = OneVsRestClassifier(LogisticRegression(max_iter=100))

        self._pipeline = Pipeline([
            ('csp', csp),
            ('classifier', clf)
        ])
        self._cv = StratifiedKFold(n_splits=5)

    def fit(self, X_train, y_train, X_test, y_test):
        # Evaluate model performance using cross-validation on the training data
        scores = cross_val_score(self._pipeline, X_train, y_train, cv=self._cv, scoring='accuracy')
        print(f'Cross-validation accuracy scores: {scores}')
        print(f'Mean accuracy: {scores.mean()}')
        self._pipeline.fit(X_train, y_train)
        # Validation test
        y_pred = self._pipeline.predict(X_test)
        print(classification_report(y_test, y_pred))

    @property
    def pipeline(self):
        return self._pipeline

def training(plot=False):
    train_module = train()

    subject_folder = "/nfs/sgoinfre/goinfre/Perso/amaach/Downloads/google/eeg-motor-movementimagery-dataset-1.0.0/files/S001"
    preprocessmodule = preprocess()
    labels, epochs = preprocessmodule.process(subject_folder, plot)
    if not plot:
        epochs_data_train = epochs.get_data()
        X_train, X_test, y_train, y_test = train_test_split(epochs_data_train, labels, test_size=0.2, random_state=42)
        model = train_module.fit(X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    training(True)