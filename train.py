from preprocessing import process, glob
from mne.decoding import CSP
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier


def fit(X_train, y_train, X_test, y_test, pipeline, cv):
    # Evaluate model performance using cross-validation on the training data
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
    print(f'Cross-validation accuracy scores: {scores}')
    print(f'Mean accuracy: {scores.mean()}')
    pipeline.fit(X_train, y_train)
    # Validation test
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))
    return pipeline


def training(plot=False):
    csp = CSP(cov_est='epoch')
    clf = OneVsRestClassifier(LogisticRegression(max_iter=100))

    pipeline = Pipeline([
        ('csp', csp),
        ('classifier', clf)
    ])
    cv = StratifiedKFold(n_splits=5)

    subject_folder = "/nfs/sgoinfre/goinfre/Perso/amaach/Downloads/google/eeg-motor-movementimagery-dataset-1.0.0/files/S001"
    labels, epochs = process(subject_folder, plot)
    if not plot:
        epochs_data_train = epochs.get_data()
        X_train, X_test, y_train, y_test = train_test_split(epochs_data_train, labels, test_size=0.2, random_state=42)
        model = fit(X_train, y_train, X_test, y_test, pipeline, cv)

if __name__ == '__main__':
    training(True)